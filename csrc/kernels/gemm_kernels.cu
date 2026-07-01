// Copyright 2024 entropy contributors
// GEMM kernels for transformer operations

#include <cuda_runtime.h>
#include <math.h>

// ============================================================
// Simple GEMM: C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
// Row-major layout
// ============================================================

__global__ void gemm_kernel(float *C, const float *A, const float *B,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// ============================================================
// GEMM with transposed B: C = A @ B^T
// A: [M, K], B: [N, K] (stored row-major), C: [M, N]
// This is the common case for linear layers: output = input @ W^T + b
// ============================================================

__global__ void gemm_transb_kernel(float *C, const float *A, const float *B,
                                    const float *bias,
                                    int M, int N, int K,
                                    bool use_bias) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[col * K + k];
    }
    if (use_bias && bias) {
        sum += bias[col];
    }
    C[row * N + col] = sum;
}

// ============================================================
// GEMM with transposed A: C = A^T @ B
// A: [K, M] (stored row-major), B: [K, N], C: [M, N]
// Used for embedding gather: output = one_hot @ embedding_table
// embedding_table: [vocab_size, hidden_size]
// ============================================================

__global__ void embed_gather_kernel(float *output, const int *input_ids,
                                     const float *embedding_table,
                                     int batch_size, int seq_len,
                                     int vocab_size, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len;

    if (idx >= total) return;

    int token_id = input_ids[idx];
    if (token_id < 0 || token_id >= vocab_size) return;

    const float *emb_row = embedding_table + token_id * hidden_size;
    float *out_row = output + idx * hidden_size;

    for (int j = 0; j < hidden_size; j++) {
        out_row[j] = emb_row[j];
    }
}

// ============================================================
// Add bias to matrix: C = A + bias[row]
// A: [M, N], bias: [N], C: [M, N]
// ============================================================

__global__ void add_bias_kernel(float *output, const float *input,
                                 const float *bias, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;

    int row = idx / N;
    output[idx] = input[idx] + bias[idx % N];
}

// ============================================================
// Element-wise activation: SwiGLU
// SwiGLU(a, b) = silu(a) * b
// a, b: [M, intermediate_size], output: [M, intermediate_size]
// ============================================================

__device__ float silu(float x) {
    float s = fminf(fmaxf(x, -12.0f), 12.0f);
    return x / (1.0f + expf(-s));
}

__global__ void swiglu_kernel(float *output, const float *a, const float *b,
                               int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    output[idx] = silu(a[idx]) * b[idx];
}

// ============================================================
// Residual add: C = A + B
// A, B, C: [M, N]
// ============================================================

__global__ void residual_add_kernel(float *output, const float *a,
                                     const float *b, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    output[idx] = a[idx] + b[idx];
}

// ============================================================
// Softmax over last dimension
// input: [M, N], output: [M, N]
// ============================================================

__global__ void softmax_kernel(float *output, const float *input,
                                int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;

    const float *row_ptr = input + row * N;
    float *out_ptr = output + row * N;

    // Find max
    float max_val = -INFINITY;
    for (int j = 0; j < N; j++) {
        if (row_ptr[j] > max_val) max_val = row_ptr[j];
    }

    // Compute sum of exp
    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        out_ptr[j] = expf(row_ptr[j] - max_val);
        sum += out_ptr[j];
    }

    // Normalize
    float inv_sum = 1.0f / (sum + 1e-8f);
    for (int j = 0; j < N; j++) {
        out_ptr[j] *= inv_sum;
    }
}

// ============================================================
// Cross-entropy loss
// logits: [M, vocab_size], labels: [M]
// Returns scalar loss per sample
// ============================================================

__global__ void cross_entropy_loss_kernel(float *loss_per_sample,
                                           const float *logits,
                                           const int *labels,
                                           int M, int vocab_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const float *logit_row = logits + row * vocab_size;
    int label = labels[row];

    // Numerical stability: subtract max
    float max_val = -INFINITY;
    for (int j = 0; j < vocab_size; j++) {
        if (logit_row[j] > max_val) max_val = logit_row[j];
    }

    float sum = 0.0f;
    for (int j = 0; j < vocab_size; j++) {
        sum += expf(logit_row[j] - max_val);
    }

    loss_per_sample[row] = -(logf(expf(logit_row[label] - max_val) / (sum + 1e-8f) + 1e-8f));
}

// ============================================================
// Compute d_logits = softmax(logits) - one_hot(labels)
// logits: [M, vocab_size], labels: [M], output: [M, vocab_size]
// ============================================================

__global__ void cross_entropy_grad_kernel(float *d_logits,
                                           const float *logits,
                                           const int *labels,
                                           int M, int vocab_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const float *logit_row = logits + row * vocab_size;
    int label = labels[row];

    // Numerical stability: subtract max
    float max_val = -INFINITY;
    for (int j = 0; j < vocab_size; j++) {
        if (logit_row[j] > max_val) max_val = logit_row[j];
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < vocab_size; j++) {
        sum_exp += expf(logit_row[j] - max_val);
    }

    float inv_sum = 1.0f / (sum_exp + 1e-8f);
    float inv_tokens = 1.0f / (float)M;
    for (int j = 0; j < vocab_size; j++) {
        float p = expf(logit_row[j] - max_val) * inv_sum;
        d_logits[row * vocab_size + j] = (p - (j == label ? 1.0f : 0.0f)) * inv_tokens;
    }
}

// ============================================================
// Scaled dot-product attention
// Q: [batch, seq_len, num_heads, head_dim]
// K: [batch, seq_len, num_kv_heads, head_dim]
// V: [batch, seq_len, num_kv_heads, head_dim]
// output: [batch, seq_len, num_heads, head_dim]
// Uses broadcasting for GQA: K, V repeated num_heads/num_kv_heads times
// ============================================================

__global__ void scaled_attention_kernel(float *output, const float *Q,
                                         const float *K, const float *V,
                                         int batch_size, int seq_len,
                                         int num_heads, int num_kv_heads,
                                         int head_dim, float softmax_scale) {
    int sample = blockIdx.z;
    int pos = blockIdx.y;
    int head = blockIdx.x;
    int dim = threadIdx.x;

    if (sample >= batch_size || pos >= seq_len || head >= num_heads || dim >= head_dim) return;

    // For GQA, map head to kv head
    int kv_head = head * num_kv_heads / num_heads;

    const float *q_row = Q + sample * seq_len * num_heads * head_dim
                           + pos * num_heads * head_dim
                           + head * head_dim;

    float max_score = -INFINITY;
    for (int j = 0; j < seq_len; j++) {
        const float *k_row = K + sample * seq_len * num_kv_heads * head_dim
                               + j * num_kv_heads * head_dim
                               + kv_head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_row[d] * k_row[d];
        }
        max_score = fmaxf(max_score, score * softmax_scale);
    }

    float sum_exp = 0.0f;
    float result = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        const float *k_row = K + sample * seq_len * num_kv_heads * head_dim
                               + j * num_kv_heads * head_dim
                               + kv_head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_row[d] * k_row[d];
        }
        float prob = expf(score * softmax_scale - max_score);
        sum_exp += prob;
        const float *v_row = V + sample * seq_len * num_kv_heads * head_dim
                               + j * num_kv_heads * head_dim
                               + kv_head * head_dim;
        result += prob * v_row[dim];
    }

    output[sample * seq_len * num_heads * head_dim
           + pos * num_heads * head_dim
           + head * head_dim + dim] = result / (sum_exp + 1e-8f);
}

// ============================================================
// RoPE: apply rotary position embeddings in-place
// q, k: [batch, seq_len, num_heads, head_dim]
// freqs: [seq_len, head_dim/2, 2] (re, im pairs)
// ============================================================

__global__ void apply_rope_kernel(float *q, float *k,
                                   int batch_size, int seq_len,
                                   int num_heads, int head_dim,
                                   const float *freqs_re, const float *freqs_im) {
    int sample = blockIdx.z;
    int pos = blockIdx.y;
    int head = blockIdx.x;
    int dim_pair = threadIdx.x;

    if (sample >= batch_size || pos >= seq_len || head >= num_heads || dim_pair >= head_dim / 2) return;

    float *q_row = q + sample * seq_len * num_heads * head_dim
                     + pos * num_heads * head_dim
                     + head * head_dim;
    float *k_row = k + sample * seq_len * num_heads * head_dim
                     + pos * num_heads * head_dim
                     + head * head_dim;

    int half_dim = head_dim / 2;
    float cos_val = freqs_re[pos * half_dim + dim_pair];
    float sin_val = freqs_im[pos * half_dim + dim_pair];

    // Q rotation
    float q0 = q_row[dim_pair];
    float q1 = q_row[dim_pair + head_dim / 2];
    q_row[dim_pair] = q0 * cos_val - q1 * sin_val;
    q_row[dim_pair + head_dim / 2] = q0 * sin_val + q1 * cos_val;

    // K rotation
    float k0 = k_row[dim_pair];
    float k1 = k_row[dim_pair + head_dim / 2];
    k_row[dim_pair] = k0 * cos_val - k1 * sin_val;
    k_row[dim_pair + head_dim / 2] = k0 * sin_val + k1 * cos_val;
}

// ============================================================
// Precompute freqs_cis for RoPE
// freqs: [max_seq_len, head_dim/2]
// base: RoPE base frequency
// ============================================================

__global__ void compute_freqs_kernel(float *freqs_re, float *freqs_im,
                                      int max_seq_len, int head_dim,
                                      float base) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= max_seq_len) return;

    int half_dim = head_dim / 2;
    float inv_freq = .0f;
    for (int i = 0; i < half_dim; i++) {
        float exponent = -(i * 2.0f) / (float)head_dim;
        float freq = powf(base, exponent);
        float angle = pos * freq;
        freqs_re[pos * half_dim + i] = cosf(angle);
        freqs_im[pos * half_dim + i] = sinf(angle);
    }
}

// ============================================================
// Host-side launch wrappers
// ============================================================

void gemm_launch(float *C, const float *A, const float *B,
                 int M, int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemm_kernel<<<grid, block, 0, stream>>>(C, A, B, M, N, K);
}

void gemm_transb_launch(float *C, const float *A, const float *B,
                         const float *bias, int M, int N, int K,
                         bool use_bias, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemm_transb_kernel<<<grid, block, 0, stream>>>(C, A, B, bias, M, N, K, use_bias);
}

void embed_gather_launch(float *output, const int *input_ids,
                          const float *embedding_table,
                          int batch_size, int seq_len,
                          int vocab_size, int hidden_size,
                          cudaStream_t stream) {
    int BLOCK = 256;
    int total = batch_size * seq_len;
    int grid = (total + BLOCK - 1) / BLOCK;
    embed_gather_kernel<<<grid, BLOCK, 0, stream>>>(
        output, input_ids, embedding_table, batch_size, seq_len, vocab_size, hidden_size);
}

void add_bias_launch(float *output, const float *input,
                      const float *bias, int M, int N,
                      cudaStream_t stream) {
    int BLOCK = 256;
    int total = M * N;
    int grid = (total + BLOCK - 1) / BLOCK;
    add_bias_kernel<<<grid, BLOCK, 0, stream>>>(output, input, bias, M, N);
}

void swiglu_launch(float *output, const float *a, const float *b,
                    int M, int N, cudaStream_t stream) {
    int BLOCK = 256;
    int total = M * N;
    int grid = (total + BLOCK - 1) / BLOCK;
    swiglu_kernel<<<grid, BLOCK, 0, stream>>>(output, a, b, M, N);
}

void residual_add_launch(float *output, const float *a, const float *b,
                          int M, int N, cudaStream_t stream) {
    int BLOCK = 256;
    int total = M * N;
    int grid = (total + BLOCK - 1) / BLOCK;
    residual_add_kernel<<<grid, BLOCK, 0, stream>>>(output, a, b, M, N);
}

void softmax_launch(float *output, const float *input,
                     int M, int N, cudaStream_t stream) {
    int BLOCK = 256;
    int grid = M;
    softmax_kernel<<<grid, BLOCK, 0, stream>>>(output, input, M, N);
}

void cross_entropy_loss_launch(float *loss_per_sample,
                                const float *logits,
                                const int *labels,
                                int M, int vocab_size,
                                cudaStream_t stream) {
    int BLOCK = 256;
    int grid = (M + BLOCK - 1) / BLOCK;
    cross_entropy_loss_kernel<<<grid, BLOCK, 0, stream>>>(
        loss_per_sample, logits, labels, M, vocab_size);
}

void cross_entropy_grad_launch(float *d_logits,
                                const float *logits,
                                const int *labels,
                                int M, int vocab_size,
                                cudaStream_t stream) {
    int BLOCK = 256;
    int grid = (M + BLOCK - 1) / BLOCK;
    cross_entropy_grad_kernel<<<grid, BLOCK, 0, stream>>>(
        d_logits, logits, labels, M, vocab_size);
}

void scaled_attention_launch(float *output, const float *Q, const float *K,
                              const float *V, int batch_size, int seq_len,
                              int num_heads, int num_kv_heads,
                              int head_dim, float softmax_scale,
                              cudaStream_t stream) {
    dim3 block(head_dim);
    dim3 grid(num_heads, seq_len, batch_size);
    scaled_attention_kernel<<<grid, block, 0, stream>>>(
        output, Q, K, V, batch_size, seq_len, num_heads, num_kv_heads,
        head_dim, softmax_scale);
}

void apply_rope_launch(float *q, float *k,
                        int batch_size, int seq_len,
                        int num_heads, int head_dim,
                        const float *freqs_re, const float *freqs_im,
                        cudaStream_t stream) {
    dim3 block(head_dim / 2);
    dim3 grid(num_heads, seq_len, batch_size);
    apply_rope_kernel<<<grid, block, 0, stream>>>(
        q, k, batch_size, seq_len, num_heads, head_dim, freqs_re, freqs_im);
}

void compute_freqs_launch(float *freqs_re, float *freqs_im,
                           int max_seq_len, int head_dim,
                           float base, cudaStream_t stream) {
    int BLOCK = 256;
    int total = max_seq_len;
    int grid = (total + BLOCK - 1) / BLOCK;
    compute_freqs_kernel<<<grid, BLOCK, 0, stream>>>(freqs_re, freqs_im, max_seq_len, head_dim, base);
}


// ============================================================
// Backward kernels
// ============================================================

__global__ void gemm_backward_a_kernel(float *dA, const float *dC, const float *B,
                                        int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= K) return;

    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        sum += dC[row * N + j] * B[j * K + col];
    }
    dA[row * K + col] = sum;
}

__global__ void gemm_backward_b_kernel(float *dB, const float *dC, const float *A,
                                        int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= K) return;

    float sum = 0.0f;
    for (int i = 0; i < M; i++) {
        sum += dC[i * N + row] * A[i * K + col];
    }
    atomicAdd(&dB[row * K + col], sum);
}

__global__ void swiglu_backward_kernel(float *d_a, float *d_b,
                                        const float *d_output,
                                        const float *a, const float *b,
                                        int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;

    float out = d_output[idx];
    float a_val = a[idx];
    float b_val = b[idx];
    float s = fminf(fmaxf(a_val, -12.0f), 12.0f);
    float silu_a = a_val / (1.0f + expf(-s));
    float sigma = 1.0f / (1.0f + expf(-s));
    float silu_deriv = sigma * (1.0f + a_val - silu_a);

    d_a[idx] = out * silu_deriv * b_val;
    d_b[idx] = out * silu_a;
}

__global__ void residual_add_backward_kernel(float *d_a, float *d_b,
                                              const float *d_c, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    if (d_a) atomicAdd(&d_a[idx], d_c[idx]);
    if (d_b) atomicAdd(&d_b[idx], d_c[idx]);
}

// ============================================================
// RMSNorm backward launch wrapper
// ============================================================

int rms_norm_backward_launch(float *d_input, float *d_weight,
                              const float *input, const float *weight,
                              const float *grad_output, int rows,
                              float eps, cudaStream_t stream) {
    return rms_norm_backward(d_input, d_weight, input, weight,
                             grad_output, rows, 1, eps, stream);
}

// ============================================================
// Backward pass launch wrappers
// ============================================================

void gemm_backward_a_launch(float *dA, const float *dC, const float *B,
                             int M, int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((K + 15) / 16, (M + 15) / 16);
    gemm_backward_a_kernel<<<grid, block, 0, stream>>>(dA, dC, B, M, N, K);
}

void gemm_backward_b_launch(float *dB, const float *dC, const float *A,
                             int M, int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((K + 15) / 16, (N + 15) / 16);
    gemm_backward_b_kernel<<<grid, block, 0, stream>>>(dB, dC, A, M, N, K);
}

void swiglu_backward_launch(float *d_a, float *d_b,
                             const float *d_output,
                             const float *a, const float *b,
                             int M, int N, cudaStream_t stream) {
    int BLOCK = 256;
    int total = M * N;
    int grid = (total + BLOCK - 1) / BLOCK;
    swiglu_backward_kernel<<<grid, BLOCK, 0, stream>>>(
        d_a, d_b, d_output, a, b, M, N);
}

void residual_add_backward_launch(float *d_a, float *d_b,
                                   const float *d_c, int M, int N,
                                   cudaStream_t stream) {
    int BLOCK = 256;
    int total = M * N;
    int grid = (total + BLOCK - 1) / BLOCK;
    residual_add_backward_kernel<<<grid, BLOCK, 0, stream>>>(d_a, d_b, d_c, M, N);
}

void attention_backward_launch(float *d_q, float *d_k, float *d_v,
                                const float *d_output, const float *q,
                                const float *k, const float *v,
                                const float *softmax_output,
                                int batch_size, int seq_len,
                                int num_heads, int num_kv_heads,
                                int head_dim, float softmax_scale,
                                cudaStream_t stream) {
    // Forward declaration from attention_kernels.cu
    extern void attention_backward(float *d_q, float *d_k, float *d_v,
                                   const float *d_output, const float *q,
                                   const float *k, const float *v,
                                   const float *softmax_output,
                                   int batch_size, int seq_len,
                                   int num_heads, int num_kv_heads,
                                   int head_dim, float softmax_scale,
                                   cudaStream_t stream);
    attention_backward(d_q, d_k, d_v, d_output, q, k, v, softmax_output,
                       batch_size, seq_len, num_heads, num_kv_heads,
                       head_dim, softmax_scale, stream);
}

void rope_backward_launch(float *d_input,
                           const float *cos, const float *sin,
                           int batch_size, int seq_len,
                           int num_heads, int head_dim,
                           cudaStream_t stream) {
    extern void rope_backward(float *d_input,
                               const float *cos, const float *sin,
                               int batch_size, int seq_len,
                               int num_heads, int head_dim,
                               cudaStream_t stream);
    rope_backward(d_input, cos, sin, batch_size, seq_len, num_heads, head_dim, stream);
}
