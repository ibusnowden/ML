// Copyright 2024 mmllm contributors
// GEMM kernel launch wrappers

#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration (defined in model.h)
typedef struct mllm_vit_block_t mllm_vit_block_t;

// GEMM: C = A @ B, A:[M,K], B:[K,N], C:[M,N]
void gemm_launch(float *C, const float *A, const float *B,
                 int M, int N, int K, cudaStream_t stream);

// GEMM with transposed B: C = A @ B^T + bias, A:[M,K], B:[N,K], C:[M,N]
void gemm_transb_launch(float *C, const float *A, const float *B,
                         const float *bias, int M, int N, int K,
                         bool use_bias, cudaStream_t stream);

// RMSNorm forward
int rms_norm_forward(float *output, const float *input,
                     const float *weight, int hidden_size, int rows,
                     float eps, cudaStream_t stream);

// RMSNorm backward
int rms_norm_backward(float *d_input, float *d_weight,
                      const float *input, const float *weight,
                      const float *grad_output, int batch_size,
                      float eps, cudaStream_t stream);

// Embedding gather: output = one_hot @ embedding_table
void embed_gather_launch(float *output, const int *input_ids,
                          const float *embedding_table,
                          int batch_size, int seq_len,
                          int vocab_size, int hidden_size,
                          cudaStream_t stream);

// Add bias: output = input + bias
void add_bias_launch(float *output, const float *input,
                      const float *bias, int M, int N,
                      cudaStream_t stream);

// SwiGLU: output = silu(a) * b
void swiglu_launch(float *output, const float *a, const float *b,
                    int M, int N, cudaStream_t stream);

// Residual add: output = a + b
void residual_add_launch(float *output, const float *a, const float *b,
                          int M, int N, cudaStream_t stream);

// Softmax over last dimension
void softmax_launch(float *output, const float *input,
                     int M, int N, cudaStream_t stream);

// Cross-entropy loss
void cross_entropy_loss_launch(float *loss_per_sample,
                                const float *logits,
                                const int *labels,
                                int M, int vocab_size,
                                cudaStream_t stream);

// Cross-entropy gradient: d_logits = softmax(logits) - one_hot(labels)
void cross_entropy_grad_launch(float *d_logits,
                                const float *logits,
                                const int *labels,
                                int M, int vocab_size,
                                cudaStream_t stream);

// Scaled dot-product attention with GQA support
void scaled_attention_launch(float *output, const float *Q, const float *K,
                              const float *V, int batch_size, int seq_len,
                              int num_heads, int num_kv_heads,
                              int head_dim, float softmax_scale,
                              cudaStream_t stream);

// RoPE in-place
void apply_rope_launch(float *q, float *k,
                        int batch_size, int seq_len,
                        int num_heads, int head_dim,
                        const float *freqs_re, const float *freqs_im,
                        cudaStream_t stream);

// Precompute RoPE freqs
void compute_freqs_launch(float *freqs_re, float *freqs_im,
                           int max_seq_len, int head_dim,
                           float base, cudaStream_t stream);

// GEMM backward: dA = dC @ B^T, dB = dC^T @ A
void gemm_backward_a_launch(float *dA, const float *dC, const float *B,
                             int M, int N, int K, cudaStream_t stream);
void gemm_backward_b_launch(float *dB, const float *dC, const float *A,
                             int M, int N, int K, cudaStream_t stream);

// SwiGLU backward
void swiglu_backward_launch(float *d_a, float *d_b,
                             const float *d_output,
                             const float *a, const float *b,
                             int M, int N, cudaStream_t stream);

// Residual add backward
void residual_add_backward_launch(float *d_a, float *d_b,
                                   const float *d_c, int M, int N,
                                   cudaStream_t stream);

// Attention backward: given d_output, Q, K, V, saved softmax(P), compute dQ, dK, dV
void attention_backward_launch(float *d_q, float *d_k, float *d_v,
                                const float *d_output, const float *q,
                                const float *k, const float *v,
                                const float *softmax_output,
                                int batch_size, int seq_len,
                                int num_heads, int num_kv_heads,
                                int head_dim, float softmax_scale,
                                cudaStream_t stream);

// RoPE backward: un-rotate gradient
void rope_backward_launch(float *d_input,
                           const float *cos, const float *sin,
                           int batch_size, int seq_len,
                           int num_heads, int head_dim,
                           cudaStream_t stream);

// Vision encoder: patch embedding conv + RMSNorm (legacy, no ViT blocks)
int vision_encoder_forward(float *output,
                            const float *image_embeds,
                            const float *conv_weight,
                            const float *norm_weight,
                            const float *norm_bias,
                            float *temp_buf,
                            int batch_size,
                            int num_images,
                            int hidden_size,
                            cudaStream_t stream);

// Full ViT vision encoder forward
// output:      [num_images, num_patches + 1, vhs]
// image_embeds:[num_images, 3 * patch_size * patch_size]
// temp_buf:    scratch space (caller allocates)
int vision_vit_forward(
    float *output,
    const float *image_embeds,
    const float *patch_conv_weight,
    const float *patch_conv_bias,
    const float *patch_norm_weight,
    const float *patch_norm_bias,
    const float *position_embeddings,
    const float *cls_token,
    const float *final_norm_weight,
    const float *final_norm_bias,
    mllm_vit_block_t *vit_blocks,
    float *temp_buf,
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
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
