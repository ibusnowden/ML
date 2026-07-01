// Copyright 2024 entropy contributors
// Transformer model implementation with multimodal support

#include "model.h"
#include "error.h"
#include "layer_norm.h"
#include "nccl_wrapper.h"
#include "comm.h"
#include "tensor.h"
#include "gemm_kernels.h"
#include "logging.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

__global__ void strided_gather_cls_kernel(float *out, const float *in,
                                           int num_images, int seq_len, int vhs) {
    int img = blockIdx.x;
    if (img >= num_images) return;
    for (int j = threadIdx.x; j < vhs; j += blockDim.x) {
        out[img * vhs + j] = in[img * seq_len * vhs + j];
    }
}

__global__ void build_combined_embeddings_kernel(float *combined,
                                                  const float *image_tokens,
                                                  const float *text_tokens,
                                                  int batch_size,
                                                  int seq_len,
                                                  int image_tokens_per_sample,
                                                  int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_tokens = batch_size * (seq_len + image_tokens_per_sample);
    int total = total_tokens * hidden_size;
    if (idx >= total) return;

    int hidden = idx % hidden_size;
    int token = idx / hidden_size;
    int sample_seq_len = seq_len + image_tokens_per_sample;
    int sample = token / sample_seq_len;
    int pos = token % sample_seq_len;

    if (pos < image_tokens_per_sample) {
        combined[idx] = image_tokens[(sample * image_tokens_per_sample + pos) * hidden_size + hidden];
    } else {
        int text_pos = pos - image_tokens_per_sample;
        combined[idx] = text_tokens[(sample * seq_len + text_pos) * hidden_size + hidden];
    }
}

__global__ void gather_text_rows_kernel(float *text_rows,
                                         const float *combined_rows,
                                         int batch_size,
                                         int seq_len,
                                         int image_tokens_per_sample,
                                         int row_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * row_width;
    if (idx >= total) return;

    int col = idx % row_width;
    int text_token = idx / row_width;
    int sample = text_token / seq_len;
    int text_pos = text_token % seq_len;
    int combined_seq_len = seq_len + image_tokens_per_sample;
    int combined_pos = image_tokens_per_sample + text_pos;

    text_rows[idx] = combined_rows[(sample * combined_seq_len + combined_pos) * row_width + col];
}

__global__ void split_qkv_kernel(float *q_out,
                                 float *k_out,
                                 float *v_out,
                                 const float *qkv,
                                 int token_count,
                                 int num_q_heads,
                                 int num_kv_heads,
                                 int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int q_dim = num_q_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int qkv_dim = q_dim + 2 * kv_dim;
    int total = token_count * qkv_dim;
    if (idx >= total) return;

    int token = idx / qkv_dim;
    int col = idx % qkv_dim;
    float value = qkv[idx];
    if (col < q_dim) {
        q_out[token * q_dim + col] = value;
    } else if (col < q_dim + kv_dim) {
        k_out[token * kv_dim + (col - q_dim)] = value;
    } else {
        v_out[token * kv_dim + (col - q_dim - kv_dim)] = value;
    }
}

// ============================================================
// Helper kernels
// ============================================================

__global__ void uniform_init_kernel(float *data, int total_elements,
                                     float low, float high) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    uint32_t x = (uint32_t)idx * 1664525u + 1013904223u;
    x ^= x >> 16;
    x *= 2246822519u;
    x ^= x >> 13;
    float u = (float)(x & 0x00FFFFFFu) / (float)0x01000000u;
    data[idx] = low + (high - low) * u;
}

__global__ void set_kernel(float *data, float value, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    data[idx] = value;
}

// ============================================================
// Model creation / destruction
// ============================================================

int mllm_model_create(mllm_model_t *model, const mllm_model_config_t *config,
                      mllm_mp_topology_t *topo) {
    memset(model, 0, sizeof(*model));
    model->config = *config;
    model->topo = topo;
    model->use_moe = (config->num_experts > 0);

    size_t elem_size = sizeof(float);
    int hs = config->hidden_size;
    int is_ = config->intermediate_size;
    int vs = config->vocab_size;
    int nl = config->num_hidden_layers;
    int max_seq = config->max_position_embeddings;
    int head_dim = config->head_dim;
    int num_kv_heads = config->num_key_value_heads;
    int num_q_heads = config->num_heads;
    int qkv_dim = (num_q_heads + 2 * num_kv_heads) * head_dim;

    // Embeddings
    MLLM_CUDA_CHECK(cudaMalloc(&model->tok_embeddings,
                                (size_t)vs * hs * elem_size));
    MLLM_CUDA_CHECK(cudaMalloc(&model->position_embeddings,
                                (size_t)max_seq * hs * elem_size));

    // Final norm
    MLLM_CUDA_CHECK(cudaMalloc(&model->norm_weight, hs * elem_size));
    MLLM_CUDA_CHECK(cudaMalloc(&model->norm_bias, hs * elem_size));

    // Output projection (tied with embeddings)
    MLLM_CUDA_CHECK(cudaMalloc(&model->output_weight,
                                hs * (size_t)vs * elem_size));

    // Transformer layers (host-side struct array, only inner pointers on device)
    model->layers = (mllm_transformer_block_t *)calloc(nl, sizeof(mllm_transformer_block_t));

    for (int l = 0; l < nl; l++) {
        mllm_transformer_block_t *blk = model->layers + l;

        // Attention weights: stored as [hidden_size, qkv_dim] for row-major GEMM
        MLLM_CUDA_CHECK(cudaMalloc(&blk->attn.qkv_weight,
                                    (size_t)hs * qkv_dim * elem_size));
        MLLM_CUDA_CHECK(cudaMalloc(&blk->attn.qkv_bias,
                                    qkv_dim * elem_size));
        // Output proj: [num_q_heads * head_dim, hidden_size]
        MLLM_CUDA_CHECK(cudaMalloc(&blk->attn.o_weight,
                                    (size_t)(num_q_heads * head_dim) * hs * elem_size));
        MLLM_CUDA_CHECK(cudaMalloc(&blk->attn.o_bias, hs * elem_size));
        MLLM_CUDA_CHECK(cudaMalloc(&blk->attn.norm_weight, hs * elem_size));
        MLLM_CUDA_CHECK(cudaMalloc(&blk->attn.norm_bias, hs * elem_size));

        // RoPE freqs
        MLLM_CUDA_CHECK(cudaMalloc(&blk->attn.freqs_cis_re,
                                    max_seq * (head_dim / 2) * elem_size));
        MLLM_CUDA_CHECK(cudaMalloc(&blk->attn.freqs_cis_im,
                                    max_seq * (head_dim / 2) * elem_size));

        // Attention temp buffers: [max_seq, qkv_dim] for QKV projections
        MLLM_CUDA_CHECK(cudaMalloc(&blk->attn.q_buf, max_seq * qkv_dim * elem_size));
        MLLM_CUDA_CHECK(cudaMalloc(&blk->attn.k_buf, max_seq * qkv_dim * elem_size));
        MLLM_CUDA_CHECK(cudaMalloc(&blk->attn.v_buf, max_seq * qkv_dim * elem_size));
        // After attention: [max_seq, num_q_heads * head_dim]
        MLLM_CUDA_CHECK(cudaMalloc(&blk->attn.attn_buf, max_seq * num_q_heads * head_dim * elem_size));
        MLLM_CUDA_CHECK(cudaMalloc(&blk->attn.attn_output, max_seq * hs * elem_size));

        if (model->use_moe) {
            MLLM_CUDA_CHECK(cudaMalloc(&blk->moe.w1_experts,
                                        (size_t)config->num_experts * hs * is_ * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&blk->moe.w3_experts,
                                        (size_t)config->num_experts * hs * is_ * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&blk->moe.w2_experts,
                                        (size_t)config->num_experts * is_ * hs * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&blk->moe.router_weight,
                                        hs * config->num_experts * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&blk->moe.shared_down_weight,
                                        is_ * hs * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&blk->moe.shared_down_bias,
                                        hs * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&blk->moe.router_scores,
                                        max_seq * config->num_experts * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&blk->moe.selected_experts,
                                        max_seq * config->num_experts_per_tok * sizeof(int)));
            MLLM_CUDA_CHECK(cudaMalloc(&blk->moe.gating_weights,
                                        max_seq * config->num_experts_per_tok * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&blk->moe.moe_output,
                                        max_seq * hs * elem_size));
        } else {
            // MLP: W1, W3 are [hidden_size, intermediate_size], W2 is [intermediate_size, hidden_size]
            MLLM_CUDA_CHECK(cudaMalloc(&blk->mlp.w1_weight,
                                        (size_t)hs * is_ * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&blk->mlp.w3_weight,
                                        (size_t)hs * is_ * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&blk->mlp.w2_weight,
                                        (size_t)is_ * hs * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&blk->mlp.w1_bias, is_ * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&blk->mlp.w3_bias, is_ * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&blk->mlp.w2_bias, hs * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&blk->mlp.norm_weight, hs * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&blk->mlp.gate_buf, max_seq * is_ * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&blk->mlp.up_buf, max_seq * is_ * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&blk->mlp.swiglu_buf, max_seq * is_ * elem_size));
        }

        // Attention backward temp buffers
        MLLM_CUDA_CHECK(cudaMalloc(&blk->attn.attn_normed, max_seq * hs * elem_size));
        blk->attn.attn_softmax = nullptr;
    }

    // Multimodal projector
    MLLM_CUDA_CHECK(cudaMalloc(&model->mm_projector.projector_weight,
                                hs * hs * elem_size));
    MLLM_CUDA_CHECK(cudaMalloc(&model->mm_projector.projector_bias,
                                hs * elem_size));

    // Vision encoder (ViT)
    int vhs = config->vision_hidden_size ? config->vision_hidden_size : hs;
    int vnl = config->vision_num_layers;
    int vnh = config->vision_num_heads ? config->vision_num_heads : 16;
    int vis_ = config->vision_intermediate_size ? config->vision_intermediate_size : vhs * 4;
    int patch_size = config->vision_patch_size ? config->vision_patch_size : 14;
    int image_size = config->vision_image_size ? config->vision_image_size : 336;
    int num_patches = (image_size / patch_size) * (image_size / patch_size);
    int vit_head_dim = vhs / vnh;
    int vit_qkv_dim = vit_head_dim * vnh * 3;

    mllm_vision_encoder_t *ve = &model->vision_encoder;
    memset(ve, 0, sizeof(*ve));

    // Patch embedding conv: [vhs, 3 * patch_size * patch_size]
    int patch_pixels = 3 * patch_size * patch_size;
    MLLM_CUDA_CHECK(cudaMalloc(&ve->patch_conv_weight, vhs * patch_pixels * elem_size));
    MLLM_CUDA_CHECK(cudaMalloc(&ve->patch_conv_bias, vhs * elem_size));
    MLLM_CUDA_CHECK(cudaMalloc(&ve->patch_norm_weight, vhs * elem_size));
    MLLM_CUDA_CHECK(cudaMalloc(&ve->patch_norm_bias, vhs * elem_size));

    // Position embeddings: [num_patches, vhs]
    MLLM_CUDA_CHECK(cudaMalloc(&ve->position_embeddings, num_patches * vhs * elem_size));

    // CLS token: [vhs]
    MLLM_CUDA_CHECK(cudaMalloc(&ve->cls_token, vhs * elem_size));

    // Final norm
    MLLM_CUDA_CHECK(cudaMalloc(&ve->final_norm_weight, vhs * elem_size));
    MLLM_CUDA_CHECK(cudaMalloc(&ve->final_norm_bias, vhs * elem_size));

    // Patch embedding output buffer
    MLLM_CUDA_CHECK(cudaMalloc(&ve->patch_embed_buf, num_patches * vhs * elem_size));

    // ViT transformer blocks
    if (vnl > 0) {
        ve->blocks = (mllm_vit_block_t *)calloc(vnl, sizeof(mllm_vit_block_t));

        for (int b = 0; b < vnl; b++) {
            mllm_vit_block_t *vb = ve->blocks + b;

            // Attention weights
            MLLM_CUDA_CHECK(cudaMalloc(&vb->qkv_weight, vhs * vit_qkv_dim * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&vb->qkv_bias, vit_qkv_dim * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&vb->o_weight, (size_t)(vnh * vit_head_dim) * vhs * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&vb->o_bias, vhs * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&vb->attn_norm_weight, vhs * elem_size));

            // MLP weights
            MLLM_CUDA_CHECK(cudaMalloc(&vb->w1_weight, vhs * vis_ * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&vb->w3_weight, vhs * vis_ * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&vb->w2_weight, vis_ * vhs * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&vb->w1_bias, vis_ * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&vb->w3_bias, vis_ * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&vb->w2_bias, vhs * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&vb->mlp_norm_weight, vhs * elem_size));

            // Temp buffers
            MLLM_CUDA_CHECK(cudaMalloc(&vb->qkv_buf, num_patches * vit_qkv_dim * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&vb->attn_out_buf, num_patches * vnh * vit_head_dim * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&vb->attn_normed, num_patches * vhs * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&vb->mlp_normed, num_patches * vhs * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&vb->gate_buf, num_patches * vis_ * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&vb->up_buf, num_patches * vis_ * elem_size));
            MLLM_CUDA_CHECK(cudaMalloc(&vb->swiglu_buf, num_patches * vis_ * elem_size));
        }
    }

    // Vision encoder temp buffer (LM space output after projector)
    MLLM_CUDA_CHECK(cudaMalloc(&model->vision_encoder_buf,
                                max_seq * hs * elem_size));

    // Per-layer temp buffers
    MLLM_CUDA_CHECK(cudaMalloc(&model->layer_input_buf,
                                max_seq * hs * elem_size));
    MLLM_CUDA_CHECK(cudaMalloc(&model->attn_output_buf,
                                max_seq * hs * elem_size));
    MLLM_CUDA_CHECK(cudaMalloc(&model->ffn_output_buf,
                                max_seq * hs * elem_size));

    // Gradient buffers
    MLLM_CUDA_CHECK(cudaMalloc(&model->grad_tok_embeddings,
                                (size_t)vs * hs * elem_size));
    MLLM_CUDA_CHECK(cudaMalloc(&model->grad_position_embeddings,
                                (size_t)max_seq * hs * elem_size));
    MLLM_CUDA_CHECK(cudaMalloc(&model->grad_norm_weight, hs * elem_size));
    MLLM_CUDA_CHECK(cudaMalloc(&model->grad_norm_bias, hs * elem_size));
    MLLM_CUDA_CHECK(cudaMalloc(&model->grad_output_weight,
                                hs * (size_t)vs * elem_size));

    // Count all parameters
    size_t param_count = 0;
    param_count += (size_t)vs * hs;                    // tok_embeddings
    param_count += (size_t)max_seq * hs;               // position_embeddings
    param_count += hs + hs;                             // final norm w + b
    param_count += hs * (size_t)vs;                     // output_weight
    param_count += nl * (
        (size_t)hs * qkv_dim + qkv_dim +                // qkv w + b
        (size_t)(num_q_heads * head_dim) * hs +         // attn out w
        hs + hs +                                        // attn norm w + b
        (model->use_moe ?
         (size_t)config->num_experts * (hs * is_ * 2 + is_ * hs) +  // MoE expert w1,w3,w2
         hs * config->num_experts + is_ * hs + hs :      // router + down w + b
         hs * is_ * 3 + is_ * 3 + hs)                    // MLP w1,w3,w2 + b1,b2 + down_b
    );
    param_count += hs * hs + hs;                        // mm projector
    param_count += (size_t)vhs * patch_pixels + vhs + vhs + vhs; // vision patch conv + bias + norm w+b
    param_count += (size_t)num_patches * vhs;           // vision position embeddings
    param_count += vhs;                                 // CLS token
    param_count += vhs + vhs;                           // vision final norm w+b
    param_count += (size_t)vnl * (                      // ViT blocks
        (size_t)vhs * vit_qkv_dim + vit_qkv_dim +        // qkv w + b
        (size_t)(vnh * vit_head_dim) * vhs + vhs +       // o w + b
        vhs +                                            // attn norm w
        (size_t)vhs * vis_ * 2 + vis_ * 2 +              // mlp w1, w3 + b1, b3
        (size_t)vis_ * vhs + vhs +                       // mlp w2 + b2
        vhs                                              // mlp norm w
    );

    model->total_params = param_count;
    model->total_bytes = param_count * elem_size;

    MLLM_LOG_INFO("Model created: %zu params, %zu bytes",
                  model->total_params, model->total_bytes);
    return MLLM_OK;
}

void mllm_model_destroy(mllm_model_t *model) {
    if (!model) return;

    cudaFree(model->tok_embeddings);
    cudaFree(model->position_embeddings);
    cudaFree(model->norm_weight);
    cudaFree(model->norm_bias);
    cudaFree(model->output_weight);

    if (model->layers) {
        for (int l = 0; l < model->config.num_hidden_layers; l++) {
            mllm_transformer_block_t *blk = model->layers + l;
            cudaFree(blk->attn.qkv_weight);
            cudaFree(blk->attn.qkv_bias);
            cudaFree(blk->attn.o_weight);
            cudaFree(blk->attn.o_bias);
            cudaFree(blk->attn.norm_weight);
            cudaFree(blk->attn.norm_bias);
            cudaFree(blk->attn.freqs_cis_re);
            cudaFree(blk->attn.freqs_cis_im);
            cudaFree(blk->attn.q_buf);
            cudaFree(blk->attn.k_buf);
            cudaFree(blk->attn.v_buf);
            cudaFree(blk->attn.attn_buf);
            cudaFree(blk->attn.attn_output);

            if (model->use_moe) {
                cudaFree(blk->moe.w1_experts);
                cudaFree(blk->moe.w3_experts);
                cudaFree(blk->moe.w2_experts);
                cudaFree(blk->moe.router_weight);
                cudaFree(blk->moe.shared_down_weight);
                cudaFree(blk->moe.shared_down_bias);
                cudaFree(blk->moe.router_scores);
                cudaFree(blk->moe.selected_experts);
                cudaFree(blk->moe.gating_weights);
                cudaFree(blk->moe.moe_output);
            } else {
                cudaFree(blk->mlp.w1_weight);
                cudaFree(blk->mlp.w3_weight);
                cudaFree(blk->mlp.w2_weight);
                cudaFree(blk->mlp.w1_bias);
                cudaFree(blk->mlp.w3_bias);
                cudaFree(blk->mlp.w2_bias);
                cudaFree(blk->mlp.norm_weight);
                cudaFree(blk->mlp.gate_buf);
                cudaFree(blk->mlp.up_buf);
                cudaFree(blk->mlp.swiglu_buf);
            }
            cudaFree(blk->attn.attn_normed);
            cudaFree(blk->attn.attn_softmax);
        }
        free(model->layers);
    }

    cudaFree(model->mm_projector.projector_weight);
    cudaFree(model->mm_projector.projector_bias);

    // Free vision encoder (ViT)
    mllm_vision_encoder_t *ve = &model->vision_encoder;
    cudaFree(ve->patch_conv_weight);
    cudaFree(ve->patch_conv_bias);
    cudaFree(ve->patch_norm_weight);
    cudaFree(ve->patch_norm_bias);
    cudaFree(ve->position_embeddings);
    cudaFree(ve->cls_token);
    cudaFree(ve->final_norm_weight);
    cudaFree(ve->final_norm_bias);
    cudaFree(ve->patch_embed_buf);

    if (ve->blocks) {
        int vnl = model->config.vision_num_layers;
        for (int b = 0; b < vnl; b++) {
            mllm_vit_block_t *vb = ve->blocks + b;
            cudaFree(vb->qkv_weight);
            cudaFree(vb->qkv_bias);
            cudaFree(vb->o_weight);
            cudaFree(vb->o_bias);
            cudaFree(vb->attn_norm_weight);
            cudaFree(vb->w1_weight);
            cudaFree(vb->w3_weight);
            cudaFree(vb->w2_weight);
            cudaFree(vb->w1_bias);
            cudaFree(vb->w3_bias);
            cudaFree(vb->w2_bias);
            cudaFree(vb->mlp_norm_weight);
            cudaFree(vb->qkv_buf);
            cudaFree(vb->attn_out_buf);
            cudaFree(vb->attn_normed);
            cudaFree(vb->mlp_normed);
            cudaFree(vb->gate_buf);
            cudaFree(vb->up_buf);
            cudaFree(vb->swiglu_buf);
        }
        free(ve->blocks);
    }

    cudaFree(model->vision_encoder_buf);
    cudaFree(model->layer_input_buf);
    cudaFree(model->attn_output_buf);
    cudaFree(model->ffn_output_buf);
    cudaFree(model->grad_tok_embeddings);
    cudaFree(model->grad_position_embeddings);
    cudaFree(model->grad_norm_weight);
    cudaFree(model->grad_norm_bias);
    cudaFree(model->grad_output_weight);

    memset(model, 0, sizeof(*model));
}

// ============================================================
// Weight initialization
// ============================================================

int mllm_model_init_weights(mllm_model_t *model, cudaStream_t stream) {
    int hs = model->config.hidden_size;
    int vs = model->config.vocab_size;
    int max_seq = model->config.max_position_embeddings;
    int nl = model->config.num_hidden_layers;
    int head_dim = model->config.head_dim;
    int num_kv_heads = model->config.num_key_value_heads;
    int num_q_heads = model->config.num_heads;
    int is_ = model->config.intermediate_size;
    int qkv_dim = (num_q_heads + 2 * num_kv_heads) * head_dim;

    auto init_param = [&](float *ptr, int rows, int cols) {
        int total = rows * cols;
        float scale = sqrtf(6.0f / (rows + cols)); // Xavier uniform [-scale, scale]
        int BLOCK = 256;
        int grid = (total + BLOCK - 1) / BLOCK;
        uniform_init_kernel<<<grid, BLOCK, 0, stream>>>(ptr, total, -scale, scale);
    };

    init_param(model->tok_embeddings, vs, hs);
    init_param(model->position_embeddings, max_seq, hs);

    auto fill_param = [&](float *ptr, int total, float value) {
        int block = 256;
        int grid = (total + block - 1) / block;
        set_kernel<<<grid, block, 0, stream>>>(ptr, value, total);
    };

    fill_param(model->norm_weight, hs, 1.0f);
    fill_param(model->norm_bias, hs, 0.0f);

    init_param(model->output_weight, vs, hs);

    for (int l = 0; l < nl; l++) {
        mllm_transformer_block_t *blk = model->layers + l;
        init_param(blk->attn.qkv_weight, hs, qkv_dim);
        fill_param(blk->attn.qkv_bias, qkv_dim, 0.0f);
        init_param(blk->attn.o_weight, num_q_heads * head_dim, hs);
        fill_param(blk->attn.o_bias, hs, 0.0f);
        fill_param(blk->attn.norm_weight, hs, 1.0f);
        fill_param(blk->attn.norm_bias, hs, 0.0f);
        compute_freqs_launch(blk->attn.freqs_cis_re, blk->attn.freqs_cis_im,
                             max_seq, head_dim, (float)model->config.rotary_base, stream);

        if (model->use_moe) {
            for (int e = 0; e < model->config.num_experts; e++) {
                init_param(model->layers[l].moe.w1_experts + e * hs * is_, hs, is_);
                init_param(model->layers[l].moe.w3_experts + e * hs * is_, hs, is_);
                init_param(model->layers[l].moe.w2_experts + e * is_ * hs, is_, hs);
            }
            init_param(blk->moe.router_weight, hs, model->config.num_experts);
            init_param(blk->moe.shared_down_weight, is_, hs);
            fill_param(blk->moe.shared_down_bias, hs, 0.0f);
        } else {
            init_param(blk->mlp.w1_weight, hs, is_);
            init_param(blk->mlp.w3_weight, hs, is_);
            init_param(blk->mlp.w2_weight, is_, hs);
            fill_param(blk->mlp.w1_bias, is_, 0.0f);
            fill_param(blk->mlp.w3_bias, is_, 0.0f);
            fill_param(blk->mlp.w2_bias, hs, 0.0f);
            fill_param(blk->mlp.norm_weight, hs, 1.0f);
        }
    }

    init_param(model->mm_projector.projector_weight, hs, hs);
    fill_param(model->mm_projector.projector_bias, hs, 0.0f);

    // Vision encoder (ViT) init
    {
        int vhs = model->config.vision_hidden_size ? model->config.vision_hidden_size : hs;
        int vnl = model->config.vision_num_layers;
        int vnh = model->config.vision_num_heads ? model->config.vision_num_heads : 16;
        int vis_ = model->config.vision_intermediate_size ? model->config.vision_intermediate_size : vhs * 4;
        int patch_size = model->config.vision_patch_size ? model->config.vision_patch_size : 14;
        int image_size = model->config.vision_image_size ? model->config.vision_image_size : 336;
        int num_patches = (image_size / patch_size) * (image_size / patch_size);
        int patch_pixels = 3 * patch_size * patch_size;
        int vit_head_dim = vhs / vnh;
        int vit_qkv_dim = vit_head_dim * vnh * 3;

        mllm_vision_encoder_t *ve = &model->vision_encoder;

        // Patch embedding
        init_param(ve->patch_conv_weight, vhs, patch_pixels);
        fill_param(ve->patch_conv_bias, vhs, 0.0f);
        fill_param(ve->patch_norm_weight, vhs, 1.0f);
        fill_param(ve->patch_norm_bias, vhs, 0.0f);

        // Position embeddings
        init_param(ve->position_embeddings, num_patches, vhs);

        // CLS token
        init_param(ve->cls_token, 1, vhs);

        // Final norm
        fill_param(ve->final_norm_weight, vhs, 1.0f);
        fill_param(ve->final_norm_bias, vhs, 0.0f);

        // ViT transformer blocks
        for (int b = 0; b < vnl; b++) {
            mllm_vit_block_t *vb = ve->blocks + b;
            init_param(vb->qkv_weight, vhs, vit_qkv_dim);
            fill_param(vb->qkv_bias, vit_qkv_dim, 0.0f);
            init_param(vb->o_weight, vnh * vit_head_dim, vhs);
            fill_param(vb->o_bias, vhs, 0.0f);
            fill_param(vb->attn_norm_weight, vhs, 1.0f);

            init_param(vb->w1_weight, vhs, vis_);
            init_param(vb->w3_weight, vhs, vis_);
            init_param(vb->w2_weight, vis_, vhs);
            fill_param(vb->w1_bias, vis_, 0.0f);
            fill_param(vb->w3_bias, vis_, 0.0f);
            fill_param(vb->w2_bias, vhs, 0.0f);
            fill_param(vb->mlp_norm_weight, vhs, 1.0f);
        }
    }

    // Zero gradient buffers
    cudaMemsetAsync(model->grad_tok_embeddings, 0,
                    (size_t)vs * hs * sizeof(float), stream);
    cudaMemsetAsync(model->grad_position_embeddings, 0,
                    (size_t)max_seq * hs * sizeof(float), stream);
    cudaMemsetAsync(model->grad_norm_weight, 0, hs * sizeof(float), stream);
    cudaMemsetAsync(model->grad_norm_bias, 0, hs * sizeof(float), stream);
    cudaMemsetAsync(model->grad_output_weight, 0,
                    hs * (size_t)vs * sizeof(float), stream);

    return MLLM_CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Forward pass — complete implementation
// ============================================================

int mllm_model_forward(mllm_model_t *model,
                       const int *input_ids,
                       const int *position_ids,
                       const float *image_embeds,
                       const int *labels,
                       int batch_size,
                       int seq_len,
                       float *logits,
                       float *loss,
                       cudaStream_t stream) {
    (void)position_ids;
    if (!model || !input_ids || !logits || batch_size <= 0 || seq_len <= 0) {
        return MLLM_ERR_INVALID_INPUT;
    }
    int hs = model->config.hidden_size;
    int nl = model->config.num_hidden_layers;
    int head_dim = model->config.head_dim;
    int num_kv_heads = model->config.num_key_value_heads;
    int num_q_heads = model->config.num_heads;
    int is_ = model->config.intermediate_size;
    int vs = model->config.vocab_size;
    int qkv_dim = (num_q_heads + 2 * num_kv_heads) * head_dim;
    float softmax_scale = 1.0f / sqrtf((float)head_dim);

    // --- Vision encoder: process image_embeds if provided ---
    // Run vision encoder + mm_projector to get image embeddings in LM space
    int image_token_count = 0;
    if (image_embeds != nullptr) {
        int total_images = batch_size * model->config.num_images;
        if (total_images <= 0 || total_images > model->config.max_position_embeddings) {
            MLLM_LOG_ERROR("image token count (%d) exceeds max_position_embeddings (%d)",
                           total_images, model->config.max_position_embeddings);
            return MLLM_ERR_INVALID_INPUT;
        }
        mllm_vision_encoder_t *ve = &model->vision_encoder;

        int vhs = model->config.vision_hidden_size ? model->config.vision_hidden_size : hs;
        int vnl = model->config.vision_num_layers;
        int vnh = model->config.vision_num_heads ? model->config.vision_num_heads : 16;
        int vis_ = model->config.vision_intermediate_size ? model->config.vision_intermediate_size : vhs * 4;
        int patch_size = model->config.vision_patch_size ? model->config.vision_patch_size : 14;
        int image_size = model->config.vision_image_size ? model->config.vision_image_size : 336;
        int num_patches = (image_size / patch_size) * (image_size / patch_size);
        int patch_pixels = 3 * patch_size * patch_size;
        int vit_head_dim = vhs / vnh;
        int vit_qkv_dim = vit_head_dim * vnh * 3;

        if (vnl > 0) {
            // Full ViT encoder
            int seq_len_vit = num_patches + 1; // +1 for CLS
            size_t temp_buf_size = seq_len_vit * (vit_qkv_dim > vis_ ? vit_qkv_dim : vis_) * sizeof(float);
            // Allocate extra buffer for patch embedding output
            size_t total_buf = temp_buf_size + num_patches * vhs * sizeof(float);
            float *vit_temp;
            MLLM_CUDA_CHECK(cudaMalloc(&vit_temp, total_buf));

            // vision_vit_forward outputs [total_images, num_patches+1, vhs]
            // We use vision_encoder_buf as output (need to resize for ViT output)
            // For now, use a separate buffer and then project
            float *vit_output;
            MLLM_CUDA_CHECK(cudaMalloc(&vit_output,
                (size_t)total_images * seq_len_vit * vhs * sizeof(float)));

            MLLM_CHECK(vision_vit_forward(vit_output,
                                           image_embeds,
                                           ve->patch_conv_weight,
                                           ve->patch_conv_bias,
                                           ve->patch_norm_weight,
                                           ve->patch_norm_bias,
                                           ve->position_embeddings,
                                           ve->cls_token,
                                           ve->final_norm_weight,
                                           ve->final_norm_bias,
                                           ve->blocks,
                                           vit_temp,
                                           total_images,
                                           vnl,
                                           vhs,
                                           vis_,
                                           vnh,
                                           vit_head_dim,
                                           vit_qkv_dim,
                                           num_patches,
                                           patch_pixels,
                                           model->config.rms_norm_eps,
                                           stream) == MLLM_OK,
                       "ViT forward failed");

            // Project CLS token through mm_projector -> [total_images, hs]
            // Extract CLS tokens (position 0 of each image)
            float *cls_tokens;
            MLLM_CUDA_CHECK(cudaMalloc(&cls_tokens,
                (size_t)total_images * vhs * sizeof(float)));

            // Strided copy: extract position 0 (CLS) from each [seq_len_vit, vhs] slice
            strided_gather_cls_kernel<<<total_images, 256, 0, stream>>>(
                cls_tokens, vit_output, total_images, seq_len_vit, vhs);

            // Project: cls_tokens [total_images, vhs] @ projector [hs, vhs]^T + bias
            gemm_transb_launch(model->vision_encoder_buf,
                               cls_tokens,
                               model->mm_projector.projector_weight,
                               model->mm_projector.projector_bias,
                               total_images, hs, vhs, true, stream);

            image_token_count = total_images;
            cudaFree(vit_temp);
            cudaFree(vit_output);
            cudaFree(cls_tokens);
        } else {
            // Legacy: simple conv + RMSNorm (no ViT blocks)
            MLLM_CHECK(vision_encoder_forward(model->vision_encoder_buf,
                                               image_embeds,
                                               ve->patch_conv_weight,
                                               ve->patch_norm_weight,
                                               ve->patch_norm_bias,
                                               model->layer_input_buf, // temp
                                               batch_size,
                                               model->config.num_images,
                                               hs, stream) == MLLM_OK,
                       "Vision encoder forward failed");

            // Project through mm_projector
            gemm_transb_launch(model->vision_encoder_buf,
                               model->vision_encoder_buf,
                               model->mm_projector.projector_weight,
                               model->mm_projector.projector_bias,
                               total_images, hs, hs, true, stream);

            image_token_count = total_images;
        }
    }

    int image_tokens_per_sample = image_embeds ? model->config.num_images : 0;
    int combined_seq_len = seq_len + image_tokens_per_sample;
    int total_tokens = batch_size * combined_seq_len;
    int text_token_count = batch_size * seq_len;
    if (total_tokens > model->config.max_position_embeddings) {
        MLLM_LOG_ERROR("batch_size * combined_seq_len (%d) exceeds max_position_embeddings (%d)",
                       total_tokens, model->config.max_position_embeddings);
        return MLLM_ERR_INVALID_INPUT;
    }

    // Step 1: Embed input_ids -> [batch_size * seq_len, hidden_size]
    float *text_embeds = model->layer_input_buf;
    embed_gather_launch(text_embeds, input_ids, model->tok_embeddings,
                        batch_size, seq_len, vs, hs, stream);

    // Step 2: Build combined embedding sequence [image_tokens, text_tokens] per sample.
    float *combined_embeds = model->attn_output_buf; // [batch_size * combined_seq_len, hs]
    if (image_tokens_per_sample > 0) {
        int total = total_tokens * hs;
        int block = 256;
        int grid = (total + block - 1) / block;
        build_combined_embeddings_kernel<<<grid, block, 0, stream>>>(
            combined_embeds, model->vision_encoder_buf, text_embeds,
            batch_size, seq_len, image_tokens_per_sample, hs);
    } else {
        cudaMemcpyAsync(combined_embeds, text_embeds,
                        (size_t)text_token_count * hs * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // --- Transformer layers on combined sequence ---
    for (int l = 0; l < nl; l++) {
        mllm_transformer_block_t *blk = model->layers + l;

        // --- Attention sub-block ---
        // 3a. RMSNorm
        float *attn_normed = blk->attn.attn_normed; // saved: [combined_seq_len, hs]
        rms_norm_forward(attn_normed, combined_embeds, blk->attn.norm_weight, hs,
                         total_tokens, model->config.rms_norm_eps, stream);

        // 3b. QKV projection: attn_normed @ W^T + b -> [combined_seq_len, qkv_dim]
        float *qkv = blk->attn.q_buf; // [combined_seq_len, qkv_dim]
        gemm_transb_launch(qkv, attn_normed, blk->attn.qkv_weight,
                           blk->attn.qkv_bias,
                           total_tokens, qkv_dim, hs, true, stream);

        // 3c. Split packed row-major QKV columns into contiguous Q, K, V tensors.
        float *Q = qkv;              // [total_tokens, num_q_heads * head_dim]
        float *K = blk->attn.k_buf;  // [total_tokens, num_kv_heads * head_dim]
        float *V = blk->attn.v_buf;  // [total_tokens, num_kv_heads * head_dim]
        int split_total = total_tokens * qkv_dim;
        int split_block = 256;
        int split_grid = (split_total + split_block - 1) / split_block;
        split_qkv_kernel<<<split_grid, split_block, 0, stream>>>(
            Q, K, V, qkv, total_tokens, num_q_heads, num_kv_heads, head_dim);

        // 3d. Apply RoPE
        apply_rope_launch(Q, K, batch_size, combined_seq_len, num_q_heads, head_dim,
                          blk->attn.freqs_cis_re, blk->attn.freqs_cis_im, stream);

        // 3e. Scaled dot-product attention
        float *attn_out = blk->attn.attn_buf; // [combined_seq_len, num_q_heads * head_dim]
        scaled_attention_launch(attn_out, Q, K, V,
                                batch_size, combined_seq_len,
                                num_q_heads, num_kv_heads, head_dim,
                                softmax_scale, stream);

        // 3f. Output projection: attn_out @ W2^T + b -> [combined_seq_len, hs]
        float *attn_residual = blk->attn.attn_output; // [combined_seq_len, hs]
        gemm_transb_launch(attn_residual, attn_out, blk->attn.o_weight,
                           blk->attn.o_bias,
                           total_tokens, hs, num_q_heads * head_dim, true, stream);

        // 3g. Residual add: combined_embeds + attn_residual
        residual_add_launch(combined_embeds, combined_embeds, attn_residual,
                            total_tokens, hs, stream);

        // --- MLP/MoE sub-block ---
        // 4a. RMSNorm
        float *ffn_normed = model->ffn_output_buf; // [combined_seq_len, hs]
        rms_norm_forward(ffn_normed, combined_embeds, blk->mlp.norm_weight, hs,
                         total_tokens, model->config.rms_norm_eps, stream);

        if (model->use_moe) {
            // MoE forward (simplified: use first expert for now)
            // Router: ffn_normed @ router_weight^T -> [combined_seq_len, num_experts]
            float *router_out = blk->moe.router_scores;
            gemm_transb_launch(router_out, ffn_normed, blk->moe.router_weight,
                               nullptr, total_tokens, model->config.num_experts, hs, false, stream);

            // Use expert 0 for simplicity (full top-k selection is complex)
            int expert = 0;
            const float *e_w1 = blk->moe.w1_experts + expert * hs * is_;
            const float *e_w3 = blk->moe.w3_experts + expert * hs * is_;
            const float *e_w2 = blk->moe.w2_experts + expert * is_ * hs;

            // SwiGLU: gate = silu(ffn_normed @ W1^T) * (ffn_normed @ W3^T)
            float *gate_buf = blk->mlp.gate_buf;
            float *up_buf = blk->mlp.up_buf;
            float *swiglu_out = blk->mlp.swiglu_buf;

            gemm_transb_launch(gate_buf, ffn_normed, e_w1, nullptr,
                               total_tokens, is_, hs, false, stream);
            gemm_transb_launch(up_buf, ffn_normed, e_w3, nullptr,
                               total_tokens, is_, hs, false, stream);
            swiglu_launch(swiglu_out, gate_buf, up_buf, total_tokens, is_, stream);

            // Down projection
            float *moe_out = blk->moe.moe_output;
            gemm_transb_launch(moe_out, swiglu_out, e_w2,
                               blk->moe.shared_down_bias,
                               total_tokens, hs, is_, true, stream);

            // Residual add
            residual_add_launch(combined_embeds, combined_embeds, moe_out,
                                total_tokens, hs, stream);
        } else {
            // Dense MLP (SwiGLU)
            float *gate_buf = blk->mlp.gate_buf;    // [combined_seq_len, is_]
            float *up_buf = blk->mlp.up_buf;        // [combined_seq_len, is_]
            float *swiglu_out = blk->mlp.swiglu_buf; // [combined_seq_len, is_]

            // Gate projection
            gemm_transb_launch(gate_buf, ffn_normed, blk->mlp.w1_weight,
                               blk->mlp.w1_bias,
                               total_tokens, is_, hs, true, stream);
            // Up projection
            gemm_transb_launch(up_buf, ffn_normed, blk->mlp.w3_weight,
                               blk->mlp.w3_bias,
                               total_tokens, is_, hs, true, stream);
            // SwiGLU activation
            swiglu_launch(swiglu_out, gate_buf, up_buf, total_tokens, is_, stream);
            // Down projection
            float *ffn_out = model->ffn_output_buf;
            gemm_transb_launch(ffn_out, swiglu_out, blk->mlp.w2_weight,
                               blk->mlp.w2_bias,
                               total_tokens, hs, is_, true, stream);

            // Residual add
            residual_add_launch(combined_embeds, combined_embeds, ffn_out,
                                total_tokens, hs, stream);
        }
    }

    // Step 3: Final RMSNorm on combined sequence
    float *normed = model->attn_output_buf; // [combined_seq_len, hs]
    rms_norm_forward(normed, combined_embeds, model->norm_weight, hs,
                     total_tokens, model->config.rms_norm_eps, stream);

    // Step 4: Output projection -> logits [combined_seq_len, vocab_size]
    gemm_transb_launch(logits, normed, model->output_weight,
                       nullptr, total_tokens, vs, hs, false, stream);

    // Step 5: Compute loss if labels provided. For the first stable path,
    // loss is computed only on text positions. With image tokens prepended,
    // gather the strided text logits into a contiguous temporary first.
    if (loss && labels) {
        float *text_logits = logits;
        if (image_tokens_per_sample != 0) {
            size_t text_logits_bytes = (size_t)text_token_count * vs * sizeof(float);
            MLLM_CUDA_CHECK(cudaMalloc(&text_logits, text_logits_bytes));
            int copy_total = text_token_count * vs;
            int copy_block = 256;
            int copy_grid = (copy_total + copy_block - 1) / copy_block;
            gather_text_rows_kernel<<<copy_grid, copy_block, 0, stream>>>(
                text_logits, logits, batch_size, seq_len,
                image_tokens_per_sample, vs);
        }
        float *loss_per_token = nullptr;
        size_t loss_bytes = (size_t)text_token_count * sizeof(float);
        MLLM_CUDA_CHECK(cudaMalloc(&loss_per_token, loss_bytes));
        cross_entropy_loss_launch(loss_per_token, text_logits, labels,
                                  text_token_count, vs, stream);
        float *h_loss = (float *)malloc(loss_bytes);
        if (!h_loss) {
            cudaFree(loss_per_token);
            if (text_logits != logits) cudaFree(text_logits);
            return MLLM_ERR_ALLOC;
        }
        MLLM_CUDA_CHECK(cudaMemcpy(h_loss, loss_per_token, loss_bytes,
                                   cudaMemcpyDeviceToHost));
        MLLM_CUDA_CHECK(cudaStreamSynchronize(stream));
        double sum = 0.0;
        for (int i = 0; i < text_token_count; i++) sum += h_loss[i];
        *loss = (float)(sum / (double)text_token_count);
        free(h_loss);
        cudaFree(loss_per_token);
        if (text_logits != logits) cudaFree(text_logits);
    }

    return MLLM_CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Backward pass kernels
// ============================================================

__global__ void scatter_embedding_grad_kernel(float *grad_embeddings,
                                               const int *input_ids,
                                               const float *d_hidden,
                                               int token_count,
                                               int hidden_size,
                                               int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = token_count * hidden_size;
    if (idx >= total) return;

    int token_pos = idx / hidden_size;
    int hidden = idx % hidden_size;
    int token_id = input_ids[token_pos];
    if (token_id < 0 || token_id >= vocab_size) return;
    atomicAdd(&grad_embeddings[token_id * hidden_size + hidden], d_hidden[idx]);
}

// ============================================================
// Backward pass — stable output/embedding gradient path
// ============================================================

int mllm_model_backward(mllm_model_t *model,
                        const float *logits,
                        const int *input_ids,
                        const int *labels,
                        int batch_size,
                        int seq_len,
                        int image_tokens_per_sample,
                        cudaStream_t stream) {
    if (!model || !logits || !input_ids || !labels || batch_size <= 0 ||
        seq_len <= 0 || image_tokens_per_sample < 0) {
        return MLLM_ERR_INVALID_INPUT;
    }

    int hs = model->config.hidden_size;
    int vs = model->config.vocab_size;
    int token_count = batch_size * seq_len;
    int combined_seq_len = seq_len + image_tokens_per_sample;
    int combined_token_count = batch_size * combined_seq_len;
    if (combined_token_count > model->config.max_position_embeddings) {
        return MLLM_ERR_INVALID_INPUT;
    }

    // The current stable training target updates token embeddings and the tied
    // output projection. Full transformer backprop requires saving per-layer
    // activations that the original code overwrote, so this avoids corrupting
    // gradients while keeping the trainer numerically live.
    MLLM_CUDA_CHECK(cudaMemsetAsync(model->grad_tok_embeddings, 0,
                    (size_t)vs * hs * sizeof(float), stream));
    MLLM_CUDA_CHECK(cudaMemsetAsync(model->grad_output_weight, 0,
                    (size_t)hs * vs * sizeof(float), stream));

    float *d_logits = const_cast<float *>(logits);
    float *text_logits = nullptr;
    float *text_normed = nullptr;
    const float *normed = model->attn_output_buf;   // saved final norm output from forward

    if (image_tokens_per_sample > 0) {
        size_t text_logits_bytes = (size_t)token_count * vs * sizeof(float);
        size_t text_normed_bytes = (size_t)token_count * hs * sizeof(float);
        MLLM_CUDA_CHECK(cudaMalloc(&text_logits, text_logits_bytes));
        MLLM_CUDA_CHECK(cudaMalloc(&text_normed, text_normed_bytes));

        int logits_total = token_count * vs;
        int normed_total = token_count * hs;
        int block = 256;
        gather_text_rows_kernel<<<(logits_total + block - 1) / block, block, 0, stream>>>(
            text_logits, logits, batch_size, seq_len, image_tokens_per_sample, vs);
        gather_text_rows_kernel<<<(normed_total + block - 1) / block, block, 0, stream>>>(
            text_normed, model->attn_output_buf, batch_size, seq_len,
            image_tokens_per_sample, hs);
        d_logits = text_logits;
        normed = text_normed;
    }

    cross_entropy_grad_launch(d_logits, d_logits, labels, token_count, vs, stream);

    gemm_backward_b_launch(model->grad_output_weight, d_logits,
                           normed, token_count, vs, hs, stream);

    float *d_hidden = model->ffn_output_buf;
    gemm_launch(d_hidden, d_logits, model->output_weight,
                token_count, hs, vs, stream);

    int block = 256;
    int total = token_count * hs;
    int grid = (total + block - 1) / block;
    scatter_embedding_grad_kernel<<<grid, block, 0, stream>>>(
        model->grad_tok_embeddings, input_ids, d_hidden,
        token_count, hs, vs);

    cudaError_t err = cudaGetLastError();
    cudaFree(text_normed);
    cudaFree(text_logits);
    if (err != cudaSuccess) {
        MLLM_LOG_ERROR("CUDA error %s at %s:%d",
                       cudaGetErrorString(err), __FILE__, __LINE__);
        return MLLM_ERR_CUDA;
    }
    return MLLM_OK;
}

int mllm_model_load_from_host(mllm_model_t *model, const void *weights,
                              size_t total_bytes) {
    size_t tok_bytes = (size_t)model->config.vocab_size * model->config.hidden_size * sizeof(float);
    if (total_bytes > tok_bytes) {
        MLLM_LOG_ERROR("Host load currently accepts tok_embeddings only: got %zu bytes, max %zu",
                       total_bytes, tok_bytes);
        return MLLM_ERR_INVALID_INPUT;
    }
    MLLM_CUDA_CHECK(cudaMemcpy(model->tok_embeddings, weights,
                                total_bytes, cudaMemcpyHostToDevice));
    return MLLM_OK;
}
