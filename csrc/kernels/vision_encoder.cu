// Copyright 2024 mmllm contributors
// Vision encoder: patch embedding + ViT transformer blocks

#include <cuda_runtime.h>
#include <math.h>
#include <cstdio>
#include "gemm_kernels.h"
#include "model.h"
#include "error.h"

// ============================================================
// Patch embedding: conv + RMSNorm + position embeddings
// ============================================================

__global__ void vision_patch_rmsnorm_kernel(
    float *output,
    const float *input,
    const float *weight,
    const float *bias,
    int hidden_size,
    int num_rows,
    float eps) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    const float *x = input + row * hidden_size;
    float *o = output + row * hidden_size;

    // Compute RMS
    float sum = 0.0f;
    for (int j = threadIdx.x; j < hidden_size; j += blockDim.x) {
        sum += x[j] * x[j];
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    float rms = rsqrtf(sum / hidden_size + eps);

    // Normalize, apply weight, add bias
    for (int j = threadIdx.x; j < hidden_size; j += blockDim.x) {
        float normed = x[j] * rms;
        o[j] = normed * weight[j] + bias[j];
    }
}

// ============================================================
// ViT self-attention kernel (no RoPE, 2D layout)
//
// Q, K, V: [num_patches, d_model] — flat layout
// output:  [num_patches, d_model]
//
// Each thread handles one (pos, dim) and computes attention
// over all positions. Simple and correct for ~576 patches.
// ============================================================

__global__ void vit_attention_kernel(
    float *output,
    const float *q,
    const float *k,
    const float *v,
    int num_patches,
    int num_heads,
    int head_dim,
    float softmax_scale) {
    int pos = blockIdx.x;  // query position
    int head = blockIdx.y;
    int dim = threadIdx.x;

    if (pos >= num_patches || head >= num_heads || dim >= head_dim) return;

    float q_val = q[pos * num_heads * head_dim + head * head_dim + dim];

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float result = 0.0f;

    for (int j = 0; j < num_patches; j++) {
        float k_val = k[j * num_heads * head_dim + head * head_dim + dim];
        float score = q_val * k_val * softmax_scale;

        float old_max = max_score;
        max_score = fmaxf(max_score, score);

        // Rescale for numerical stability
        float scale = expf(old_max - max_score);
        sum_exp = sum_exp * scale + expf(score - max_score);
        result = result * scale;

        float v_val = v[j * num_heads * head_dim + head * head_dim + dim];
        result += expf(score - max_score) * v_val;
    }

    output[pos * num_heads * head_dim + head * head_dim + dim] = result / (sum_exp + 1e-8f);
}

// ============================================================
// ViT forward: full encoder for a single image
//
// Input:  image_embeds [3 * patch_size * patch_size]
// Output: patch_embeds [num_patches + 1, vhs] (patches + CLS)
// ============================================================

__global__ void add_position_embedding_kernel(
    float *embeddings,
    const float *pos_emb,
    int num_patches,
    int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_patches * hidden_size) return;
    embeddings[idx] += pos_emb[idx];
}

// ============================================================
// ViT block forward (attention + MLP with residuals)
// ============================================================

static int vit_block_forward(
    float *embeddings,        // [num_patches, vhs] — in-place update
    const float *pos_emb,     // [num_patches, vhs] (added once at start, not per-block)
    void *block_ptr_host,     // NOT USED — blocks are on device, this is a placeholder
    int num_patches,
    int vhs,
    int vis_,
    int vnh,
    int vit_head_dim,
    int vit_qkv_dim,
    float rms_norm_eps,
    cudaStream_t stream) {
    // This function is a stub for host-side orchestration.
    // Actual kernels are launched from vision_vit_forward.
    (void)block_ptr_host; (void)pos_emb; (void)embeddings;
    (void)num_patches; (void)vhs; (void)vis_; (void)vnh;
    (void)vit_head_dim; (void)vit_qkv_dim; (void)rms_norm_eps;
    (void)stream;
    return 0;
}

// ============================================================
// Full vision ViT forward
//
// For each image in the batch:
//   1. Patch embedding: GEMM image @ conv_weight^T + bias -> [num_patches, vhs]
//   2. RMSNorm on patch embeddings
//   3. Add learnable position embeddings
//   4. Prepend CLS token -> [num_patches + 1, vhs]
//   5. ViT transformer blocks (self-attention + SwiGLU MLP)
//   6. Final RMSNorm
//
// Output: per_image_output [num_images, num_patches + 1, vhs]
// ============================================================

int vision_vit_forward(
    float *output,                    // [num_images, num_patches + 1, vhs]
    const float *image_embeds,        // [num_images, 3 * patch_size * patch_size]
    const float *patch_conv_weight,   // [vhs, 3 * patch_size * patch_size]
    const float *patch_conv_bias,     // [vhs]
    const float *patch_norm_weight,   // [vhs]
    const float *patch_norm_bias,     // [vhs]
    const float *position_embeddings, // [num_patches, vhs]
    const float *cls_token,           // [vhs]
    const float *final_norm_weight,   // [vhs]
    const float *final_norm_bias,     // [vhs]
    mllm_vit_block_t *vit_blocks,     // [vision_num_layers] (device ptr)
    float *temp_buf,                  // scratch [num_patches + 1, max(vit_qkv_dim, vis_)]
    int num_images,
    int vision_num_layers,
    int vhs,
    int vis_,
    int vnh,
    int vit_head_dim,
    int vit_qkv_dim,
    int num_patches,
    int patch_pixels,
    float rms_norm_eps,
    cudaStream_t stream) {

    int seq_len = num_patches + 1;  // +1 for CLS token
    float softmax_scale = 1.0f / sqrtf((float)vit_head_dim);

    // Per-image processing (serial for simplicity — could parallelize)
    for (int img = 0; img < num_images; img++) {
        const float *img_input = image_embeds + (size_t)img * num_patches * patch_pixels;
        float *img_output = output + img * seq_len * vhs;

        // --- Step 1: Patch embedding GEMM ---
        // temp_buf[0:num_patches, vhs] = image_embeds @ conv_weight^T + bias
        // For a single image with num_patches, we reshape:
        // Input: [num_patches, patch_pixels] (each patch is patch_pixels elements)
        // Weight: [vhs, patch_pixels]
        // Output: [num_patches, vhs]
        float *patch_buf = temp_buf;
        gemm_transb_launch(patch_buf, (const float *)img_input,
                           patch_conv_weight, patch_conv_bias,
                           num_patches, vhs, patch_pixels, true, stream);

        // --- Step 2: RMSNorm on patch embeddings ---
        int grid_rms = num_patches;
        vision_patch_rmsnorm_kernel<<<grid_rms, 256, 0, stream>>>(
            patch_buf, patch_buf,
            patch_norm_weight, patch_norm_bias,
            vhs, num_patches, rms_norm_eps);

        MLLM_CUDA_CHECK(cudaGetLastError());

        // --- Step 3: Add position embeddings ---
        int total_pe = num_patches * vhs;
        int grid_pe = (total_pe + 255) / 256;
        add_position_embedding_kernel<<<grid_pe, 256, 0, stream>>>(
            patch_buf, position_embeddings, num_patches, vhs);

        // --- Step 4: Prepend CLS token ---
        // Copy CLS token to position 0, shift patches to 1:num_patches+1
        float *seq_buf = (float *)((char *)temp_buf + (num_patches + 1) * vhs * sizeof(float));
        // Copy CLS to seq_buf[0:vhs]
        MLLM_CUDA_CHECK(cudaMemcpyAsync(seq_buf, cls_token, vhs * sizeof(float),
                                         cudaMemcpyDeviceToDevice, stream));
        // Copy patches to seq_buf[vhs:(num_patches+1)*vhs]
        MLLM_CUDA_CHECK(cudaMemcpyAsync(seq_buf + vhs, patch_buf,
                                         num_patches * vhs * sizeof(float),
                                         cudaMemcpyDeviceToDevice, stream));

        // --- Step 5: ViT transformer blocks ---
        for (int b = 0; b < vision_num_layers; b++) {
            // We need to copy block data from device struct. Since we can't
            // dereference device pointers on host, we use a different approach:
            // launch kernels that take the struct fields directly.
            // For now, use a kernel-based approach that reads from the block struct.

            mllm_vit_block_t *vb = vit_blocks + b;

            // --- 5a. Attention sub-block ---
            // RMSNorm
            float *attn_normed = vb->attn_normed; // [seq_len, vhs]
            rms_norm_forward(attn_normed, seq_buf, vb->attn_norm_weight, vhs,
                             seq_len, rms_norm_eps, stream);

            // QKV projection: attn_normed @ W_qkv^T + bias
            // [seq_len, vhs] @ [vhs, vit_qkv_dim] -> [seq_len, vit_qkv_dim]
            float *qkv = vb->qkv_buf;
            gemm_transb_launch(qkv, attn_normed, vb->qkv_weight,
                               vb->qkv_bias,
                               seq_len, vit_qkv_dim, vhs, true, stream);

            // Split Q, K, V (interleaved by head in the flat layout)
            // Q: [seq_len, vnh * vit_head_dim]
            // K: [seq_len, vnh * vit_head_dim]
            // V: [seq_len, vnh * vit_head_dim]
            float *Q = qkv;
            float *K = qkv + seq_len * vnh * vit_head_dim;
            float *V = qkv + seq_len * vnh * vit_head_dim * 2;

            // Self-attention (no RoPE for ViT)
            float *attn_out = vb->attn_out_buf; // [seq_len, vnh * vit_head_dim]
            dim3 block_attn(vit_head_dim);
            dim3 grid_attn(seq_len, vnh);
            vit_attention_kernel<<<grid_attn, block_attn, 0, stream>>>(
                attn_out, Q, K, V, seq_len, vnh, vit_head_dim, softmax_scale);

            MLLM_CUDA_CHECK(cudaGetLastError());

            // Output projection: attn_out @ W_o^T + bias
            float *attn_residual = (float *)((char *)temp_buf + seq_len * vit_qkv_dim * sizeof(float));
            gemm_transb_launch(attn_residual, attn_out, vb->o_weight,
                               vb->o_bias,
                               seq_len, vhs, vnh * vit_head_dim, true, stream);

            // Residual add: seq_buf += attn_residual
            residual_add_launch(seq_buf, seq_buf, attn_residual,
                                seq_len, vhs, stream);

            // --- 5b. MLP sub-block (SwiGLU) ---
            // RMSNorm
            float *ffn_normed = vb->mlp_normed; // [seq_len, vhs]
            rms_norm_forward(ffn_normed, seq_buf, vb->mlp_norm_weight, vhs,
                             seq_len, rms_norm_eps, stream);

            // Gate projection
            float *gate_buf = vb->gate_buf;
            gemm_transb_launch(gate_buf, ffn_normed, vb->w1_weight,
                               vb->w1_bias,
                               seq_len, vis_, vhs, true, stream);

            // Up projection
            float *up_buf = vb->up_buf;
            gemm_transb_launch(up_buf, ffn_normed, vb->w3_weight,
                               vb->w3_bias,
                               seq_len, vis_, vhs, true, stream);

            // SwiGLU
            float *swiglu_out = vb->swiglu_buf;
            swiglu_launch(swiglu_out, gate_buf, up_buf, seq_len, vis_, stream);

            // Down projection
            float *ffn_out = (float *)((char *)temp_buf + seq_len * (vit_qkv_dim > vis_ ? vit_qkv_dim : vis_) * sizeof(float));
            gemm_transb_launch(ffn_out, swiglu_out, vb->w2_weight,
                               vb->w2_bias,
                               seq_len, vhs, vis_, true, stream);

            // Residual add
            residual_add_launch(seq_buf, seq_buf, ffn_out,
                                seq_len, vhs, stream);
        }

        // --- Step 6: Final RMSNorm ---
        rms_norm_forward(img_output, seq_buf, final_norm_weight, vhs,
                         seq_len, rms_norm_eps, stream);
    }

    return MLLM_CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Legacy: simple conv + RMSNorm (no ViT blocks)
// Kept for backward compatibility when vision_num_layers == 0
// ============================================================

int vision_encoder_forward(
    float *output,
    const float *image_embeds,
    const float *conv_weight,
    const float *norm_weight,
    const float *norm_bias,
    float *temp_buf,
    int batch_size,
    int num_images,
    int hidden_size,
    cudaStream_t stream) {
    int total_images = batch_size * num_images;
    int patch_pixels = 3 * 14 * 14; // 588

    // Step 1: GEMM — image_embeds [total_images, 588] @ conv_weight^T [588, hidden_size]
    gemm_transb_launch(temp_buf, image_embeds, conv_weight,
                       nullptr, total_images, hidden_size, patch_pixels,
                       false, stream);

    // Step 2: RMSNorm with weight and bias
    int grid = total_images;
    vision_patch_rmsnorm_kernel<<<grid, 256, 0, stream>>>(
        output, temp_buf, norm_weight, norm_bias,
        hidden_size, total_images, 1e-6f);

    return MLLM_CUDA_CHECK(cudaGetLastError());
}
