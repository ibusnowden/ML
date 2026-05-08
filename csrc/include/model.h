// Copyright 2024 mmllm contributors
// Transformer model definition with multimodal support

#pragma once

#include "tensor.h"
#include "nccl_wrapper.h"
#include "comm.h"
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Model configuration
typedef struct {
    int hidden_size;       // d_model
    int intermediate_size; // d_ffn
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;   // for GQA
    int num_heads;             // num_attention_heads for attention
    int head_dim;
    int max_position_embeddings;
    int vocab_size;
    int image_size;            // for multimodal
    int num_images;            // max images per sample
    int num_experts;           // MoE: 0 = dense, >0 = MoE
    int num_experts_per_tok;   // top-k for MoE
    float rms_norm_eps;
    float attention_dropout;
    int rotary_base;           // RoPE base frequency

    // Vision (ViT) config
    int vision_hidden_size;    // ViT hidden dim (often same as hidden_size)
    int vision_num_layers;     // ViT transformer blocks (0 = no ViT, just conv)
    int vision_num_heads;      // ViT attention heads
    int vision_intermediate_size; // ViT MLP intermediate size
    int vision_patch_size;     // Patch size (default 14)
    int vision_image_size;     // Input image size in pixels (default 336)
} mllm_model_config_t;

// RMSNorm (used instead of LayerNorm in modern transformers)
int rms_norm_forward(float *output, const float *input,
                     const float *weight, int hidden_size, int rows,
                     float eps, cudaStream_t stream);

int rms_norm_backward(float *d_input, float *d_weight,
                      const float *input, const float *weight,
                      const float *grad_output, int hidden_size, int rows,
                      float eps, cudaStream_t stream);

// Attention block (grouped-query attention)
typedef struct {
    float *qkv_weight;   // [hidden_size, (num_q_heads + 2*num_kv_heads) * head_dim]
    float *qkv_bias;     // [(num_q_heads + 2*num_kv_heads) * head_dim]
    float *o_weight;     // [num_q_heads * head_dim, hidden_size]
    float *o_bias;       // [hidden_size]
    float *norm_weight;  // [hidden_size]
    float *norm_bias;    // [hidden_size]

    // Precomputed freqs_cis for RoPE: [max_seq_len, num_heads, head_dim/2, 2]
    float *freqs_cis_re;
    float *freqs_cis_im;

    // Temp buffers for attention
    float *q_buf;
    float *k_buf;
    float *v_buf;
    float *attn_buf;
    float *attn_output;

    // Temp buffers for backward pass (saved intermediates)
    float *attn_normed;   // input to attention sub-block, [max_seq, hidden_size]
    float *attn_softmax;  // saved softmax output P, [max_seq, max_seq, num_heads]
} mllm_attention_t;

// MLP block (SwiGLU)
typedef struct {
    float *w1_weight;  // [hidden_size, intermediate_size]  (gate)
    float *w3_weight;  // [hidden_size, intermediate_size]  (up)
    float *w2_weight;  // [intermediate_size, hidden_size]   (down)
    float *w1_bias;    // [intermediate_size]
    float *w3_bias;    // [intermediate_size]
    float *w2_bias;    // [hidden_size]
    float *norm_weight;// [hidden_size]

    // Temp buffers
    float *gate_buf;
    float *up_buf;
    float *swiglu_buf;
} mllm_mlp_t;

// MoE layer (replaces MLP when num_experts > 0)
typedef struct {
    // Expert weights: [num_experts, hidden_size, intermediate_size] for each weight
    float *w1_experts; // [num_experts, hidden_size, intermediate_size]
    float *w3_experts; // [num_experts, hidden_size, intermediate_size]
    float *w2_experts; // [num_experts, intermediate_size, hidden_size]

    // Shared router and down projection
    float *router_weight; // [hidden_size, num_experts]
    float *shared_down_weight; // [intermediate_size, hidden_size]
    float *shared_down_bias;   // [hidden_size]

    // MoE temps
    float *router_scores;  // [batch_size * seq_len, num_experts]
    int *selected_experts; // [batch_size * seq_len, top_k]
    float *gating_weights; // [batch_size * seq_len, top_k]
    float *moe_output;     // [batch_size * seq_len, hidden_size]
} mllm_moe_t;

// Transformer decoder layer
typedef struct {
    mllm_attention_t attn;
    union {
        mllm_mlp_t mlp;
        mllm_moe_t moe;
    };
    int use_moe;
} mllm_transformer_block_t;

// Multimodal projector (connects vision encoder to language model)
typedef struct {
    float *projector_weight; // [hidden_size, vision_hidden_size]
    float *projector_bias;   // [hidden_size]
} mllm_mm_projector_t;

// ViT transformer block (attention + MLP, no residual norm)
typedef struct mllm_vit_block_t {
    // Attention weights
    float *qkv_weight;   // [vision_hidden_size, (vision_num_heads * 3) * vit_head_dim]
    float *qkv_bias;     // [(vision_num_heads * 3) * vit_head_dim]
    float *o_weight;     // [vision_num_heads * vit_head_dim, vision_hidden_size]
    float *o_bias;       // [vision_hidden_size]
    float *attn_norm_weight; // [vision_hidden_size]

    // MLP (SwiGLU)
    float *w1_weight;    // [vision_hidden_size, vision_intermediate_size]
    float *w3_weight;    // [vision_hidden_size, vision_intermediate_size]
    float *w2_weight;    // [vision_intermediate_size, vision_hidden_size]
    float *w1_bias;      // [vision_intermediate_size]
    float *w3_bias;      // [vision_intermediate_size]
    float *w2_bias;      // [vision_hidden_size]
    float *mlp_norm_weight; // [vision_hidden_size]

    // Temp buffers
    float *qkv_buf;      // [num_patches, (3 * vision_num_heads) * vit_head_dim]
    float *attn_out_buf; // [num_patches, vision_num_heads * vit_head_dim]
    float *attn_normed;  // [num_patches, vision_hidden_size] (saved for backward)
    float *mlp_normed;   // [num_patches, vision_hidden_size]
    float *gate_buf;     // [num_patches, vision_intermediate_size]
    float *up_buf;       // [num_patches, vision_intermediate_size]
    float *swiglu_buf;   // [num_patches, vision_intermediate_size]
} mllm_vit_block_t;

// Full vision encoder (ViT)
typedef struct {
    // Patch embedding
    float *patch_conv_weight; // [vision_hidden_size, 3 * patch_size * patch_size]
    float *patch_conv_bias;   // [vision_hidden_size]
    float *patch_norm_weight; // [vision_hidden_size]
    float *patch_norm_bias;   // [vision_hidden_size]

    // Learnable position embeddings
    float *position_embeddings; // [num_patches, vision_hidden_size]

    // CLS token embedding
    float *cls_token;           // [vision_hidden_size]

    // ViT transformer blocks
    mllm_vit_block_t *blocks;  // [vision_num_layers]

    // Final norm
    float *final_norm_weight;  // [vision_hidden_size]
    float *final_norm_bias;    // [vision_hidden_size]

    // Temp buffer for patch embedding output
    float *patch_embed_buf;    // [num_patches, vision_hidden_size]
} mllm_vision_encoder_t;

// Full model
typedef struct {
    mllm_model_config_t config;
    mllm_mp_topology_t *topo;
    int use_moe;

    // Embedding
    float *tok_embeddings;     // [vocab_size, hidden_size]
    float *position_embeddings;// [max_position_embeddings, hidden_size]

    // Final RMS norm
    float *norm_weight;
    float *norm_bias;

    // Output projection (tied embeddings)
    float *output_weight;      // [vocab_size, hidden_size] for GEMM transposed-B

    // Transformer layers
    mllm_transformer_block_t *layers;

    // Multimodal
    mllm_mm_projector_t mm_projector;

    // Vision encoder (ViT)
    mllm_vision_encoder_t vision_encoder;

    // Vision encoder temp buffers
    float *vision_encoder_buf;    // [total_images, hidden_size] (LM space, after projector)
    float *vit_temp_buf;          // For ViT internal computations
    float *vit_output_buf;        // [total_images, seq_len_vit, vhs]
    float *cls_tokens_buf;        // [total_images, vhs]

    // Per-layer temp buffers (for activations during forward/backward)
    float *layer_input_buf;    // [batch_size * seq_len, hidden_size]
    float *attn_output_buf;    // [batch_size * seq_len, hidden_size]
    float *ffn_output_buf;     // [batch_size * seq_len, hidden_size]

    // Gradient buffers (for accumulation)
    float *grad_tok_embeddings;
    float *grad_position_embeddings;
    float *grad_norm_weight;
    float *grad_norm_bias;
    float *grad_output_weight;

    // Total parameter count and size in bytes
    size_t total_params;
    size_t total_bytes;
} mllm_model_t;

// Model creation and destruction
int mllm_model_create(mllm_model_t *model, const mllm_model_config_t *config,
                      mllm_mp_topology_t *topo);
void mllm_model_destroy(mllm_model_t *model);

// Forward pass
// input_ids:     [batch_size, seq_len]
// position_ids:  [batch_size, seq_len]
// image_embeds:  device patch-major float32 image data (optional, NULL if text-only)
// labels:        [batch_size, seq_len] (optional, NULL if no labels for loss)
int mllm_model_forward(mllm_model_t *model,
                       const int *input_ids,
                       const int *position_ids,
                       const float *image_embeds,
                       const int *labels,
                       int batch_size,
                       int seq_len,
                       float *logits,
                       float *loss,
                       cudaStream_t stream);

// Backward pass
int mllm_model_backward(mllm_model_t *model,
                        const float *logits,
                        const int *input_ids,
                        const int *labels,
                        int batch_size,
                        int seq_len,
                        int image_tokens_per_sample,
                        cudaStream_t stream);

// Initialize weights (Xavier uniform)
int mllm_model_init_weights(mllm_model_t *model, cudaStream_t stream);

// Copy model weights from host (for loading pretrained weights)
int mllm_model_load_from_host(mllm_model_t *model, const void *weights,
                              size_t total_bytes);

#ifdef __cplusplus
}
#endif
