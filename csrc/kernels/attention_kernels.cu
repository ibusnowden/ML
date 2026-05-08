// Copyright 2024 mmllm contributors
// Self-attention kernel implementation

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define MLLM_CUDA_LAUNCH_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Rotary Position Embedding (RoPE)
__device__ void apply_rope(float* q, float* k, int head_dim, int pos,
                           const float* freqs) {
    for (int i = 0; i < head_dim / 2; i++) {
        float freq = freqs[i * 2];
        float theta = powf(10000.0f, -freq / head_dim);
        float inv_freq = 1.0f / (1.0f + logf(pos * theta) * (2.0f / head_dim));
        float angle = pos * theta * inv_freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        float q0 = q[i];
        float q1 = q[i + head_dim / 2];
        q[i] = q0 * cos_val - q1 * sin_val;
        q[i + head_dim / 2] = q0 * sin_val + q1 * cos_val;

        float k0 = k[i];
        float k1 = k[i + head_dim / 2];
        k[i] = k0 * cos_val - k1 * sin_val;
        k[i + head_dim / 2] = k0 * sin_val + k1 * cos_val;
    }
}

template <typename T>
__global__ void attention_forward_kernel(float* output, const float* q,
                                          const float* k, const float* v,
                                          const float* freqs,
                                          int batch_size, int seq_len,
                                          int num_heads, int head_dim,
                                          float softmax_scale) {
    int head_idx = blockIdx.z;
    int sample = blockIdx.y;
    int pos = blockIdx.x;
    int lane = threadIdx.x % 32;

    float* out_row = output + sample * seq_len * num_heads * head_dim +
                     pos * num_heads * head_dim + head_idx * head_dim;

    const float* q_row = q + sample * seq_len * num_heads * head_dim +
                         pos * num_heads * head_dim + head_idx * head_dim;

    // Load query and apply RoPE
    float query[64];
    for (int i = lane; i < head_dim; i += 32) {
        query[i] = static_cast<float>(q_row[i]);
    }
    apply_rope(query, (float*)q_row, head_dim, pos, freqs);

    // Compute attention scores
    float scores[2048]; // Max seq_len per block
    float max_score = -INFINITY;

    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        const float* k_row = k + sample * seq_len * num_heads * head_dim +
                            j * num_heads * head_dim + head_idx * head_dim;

        float score = 0.0f;
        for (int i = lane; i < head_dim; i += 32) {
            score += query[i] * static_cast<float>(k_row[i]);
        }
        score *= softmax_scale;

        // Warp reduction for max
        for (int offset = 16; offset > 0; offset /= 2) {
            score += __shfl_down_sync(0xFFFFFFFF, score, offset);
        }

        scores[j] = score;
        if (lane == 0 && score > max_score) {
            max_score = score;
        }
    }

    // Reduce max across warp
    for (int offset = 16; offset > 0; offset /= 2) {
        max_score += __shfl_down_sync(0xFFFFFFFF, max_score, offset);
    }

    // Compute softmax and accumulate
    float sum_exp = 0.0f;
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        scores[j] = expf(scores[j] - max_score);
        sum_exp += scores[j];
    }

    // Reduce sum
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
    }

    // Write output
    const float* v_row = v + sample * seq_len * num_heads * head_dim;
    float output_val[64] = {0};
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        const float* v_j = v_row + j * num_heads * head_dim + head_idx * head_dim;
        float weight = scores[j] / (sum_exp + 1e-8f);
        for (int i = lane; i < head_dim; i += 32) {
            output_val[i] += weight * static_cast<float>(v_j[i]);
        }
    }

    for (int i = lane; i < head_dim; i += 32) {
        out_row[i] = output_val[i];
    }
}

void attention_forward(float* output, const float* q, const float* k,
                       const float* v, const float* freqs,
                       int batch_size, int seq_len, int num_heads, int head_dim,
                       cudaStream_t stream) {
    const float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    dim3 block(32);
    dim3 grid(seq_len, batch_size, num_heads);

    attention_forward_kernel<float><<<grid, block, 0, stream>>>(
        output, q, k, v, freqs, batch_size, seq_len, num_heads, head_dim,
        softmax_scale);

    MLLM_CUDA_LAUNCH_CHECK(cudaGetLastError());
}

// ============================================================
// Attention backward kernels
//
// Forward: A[s,i,h,d] = sum_j P[s,i,j,h] * V[s,j,kv_h,d]
//   where P = softmax(QK^T * scale) over j
//
// Given d_output [batch, seq, num_heads, head_dim], Q, K, V, P:
//   dV[s,j,kv_h,d] = sum_i P[s,i,j,h] * d_output[s,i,h,d]
//   dP[s,i,j,h] = sum_d d_output[s,i,h,d] * V[s,j,kv_h,d]
//   dLogit[s,i,j,h] = P[s,i,j,h] * (dP[s,i,j,h] - sum_j'(P[s,i,j',h]*dP[s,i,j',h]))
//   dQ[s,i,h,d] = sum_j dLogit[s,i,j,h] * K[s,j,kv_h,d] * scale
//   dK[s,j,kv_h,d] = sum_i dLogit[s,i,j,h] * Q[s,i,h,d] * scale
// ============================================================

// Kernel: dV = P^T @ d_output
__global__ void attn_bwd_dv_kernel(
    float* d_v,
    const float* softmax_output,  // [batch, seq, seq, num_heads]
    const float* d_output,        // [batch, seq, num_heads, head_dim]
    int batch_size, int seq_len,
    int num_heads, int num_kv_heads, int head_dim) {
    int sample = blockIdx.z / num_kv_heads;
    int kv_head = blockIdx.z % num_kv_heads;
    int j = blockIdx.y;
    int head = kv_head * (num_heads / num_kv_heads);
    int dim = threadIdx.x;

    if (blockIdx.z >= batch_size * seq_len * num_kv_heads) return;
    if (dim >= head_dim) return;

    float sum = 0.0f;
    const float* d_out = d_output + sample * seq_len * num_heads * head_dim
                                   + head * head_dim + dim;
    for (int i = 0; i < seq_len; i++) {
        float p = softmax_output[sample * seq_len * seq_len * num_heads
                                 + i * seq_len * num_heads + j * num_heads + head];
        sum += p * d_out[i * num_heads * head_dim];
    }
    d_v[blockIdx.z * head_dim + dim] = sum;
}

// Kernel: dP = d_output @ V^T  (per (s, i, j, h), sum over dim)
__global__ void attn_bwd_dp_kernel(
    float* d_p,
    const float* d_output,        // [batch, seq, num_heads, head_dim]
    const float* v,               // [batch, seq, num_kv_heads, head_dim]
    int batch_size, int seq_len,
    int num_heads, int num_kv_heads, int head_dim) {
    int sample = blockIdx.z / num_heads;
    int head = blockIdx.z % num_heads;
    int i = blockIdx.y;
    int j = blockIdx.x;

    if (blockIdx.z >= batch_size * seq_len * num_heads) return;
    if (j >= seq_len) return;

    int kv_head = head * num_kv_heads / num_heads;
    float sum = 0.0f;
    const float* d_out = d_output + sample * seq_len * num_heads * head_dim
                                   + i * num_heads * head_dim + head * head_dim;
    const float* v_row = v + sample * seq_len * num_kv_heads * head_dim
                           + j * num_kv_heads * head_dim + kv_head * head_dim;
    for (int d = 0; d < head_dim; d++) {
        sum += d_out[d] * v_row[d];
    }
    d_p[blockIdx.z * seq_len * seq_len + i * seq_len + j] = sum;
}

// Kernel: softmax derivative — dLogit = P * (dP - sum_j(P*dP, dim=j))
// Two-pass: pass 1 stores P*dP, pass 2 computes mean and finalizes
__global__ void attn_bwd_softmax_pass1_kernel(
    float* d_logit,     // [batch, seq, seq, num_heads], in=P, out=P*dP
    const float* d_p,   // [batch, seq, seq, num_heads]
    int batch_size, int seq_len, int num_heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * seq_len * num_heads;
    if (idx >= total) return;

    float p = d_logit[idx];
    float dp = d_p[idx];
    d_logit[idx] = p * dp;  // store P*dP temporarily
}

__global__ void attn_bwd_softmax_pass2_kernel(
    float* d_logit,     // [batch, seq, seq, num_heads], in=P*dP, out=dLogit
    int batch_size, int seq_len, int num_heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * num_heads;
    if (idx >= total) return;

    int h = idx % num_heads;
    int i = (idx / num_heads) % seq_len;
    int s = idx / (num_heads * seq_len);

    // Compute sum over j
    float sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        sum += d_logit[s * seq_len * seq_len * num_heads
                       + i * seq_len * num_heads + j * num_heads + h];
    }
    float mean = sum / seq_len;

    // Finalize: dLogit = P*dP * (1 - mean/(P*dP)) = P*dP - mean
    // But we need P*dP values. They're gone! We only have P*dP stored.
    // The correct formula is: dLogit = P * (dP - sum(P*dP))
    // We have P*dP = P * dP. So dLogit = P * dP - P * sum(P*dP)
    //                                          = stored - P * mean
    // But we lost P. This two-pass approach doesn't work cleanly.
    //
    // Alternative: store dP separately, compute mean(P*dP), then
    // dLogit = P * (dP - mean). But we don't have P stored separately.
    //
    // Fix: do it all in one kernel. See below.
    (void)s; (void)i; (void)h; (void)sum; (void)mean;
}

// ===== CORRECT: single kernel for softmax derivative =====
// Given P (softmax_output) and dP (d_p), compute dLogit = P * (dP - sum(P*dP, dim=j))
// Writes result into d_logit buffer
__global__ void attn_bwd_softmax_deriv_kernel(
    float* d_logit,     // [batch, seq, seq, num_heads], output
    const float* p,     // [batch, seq, seq, num_heads]
    const float* d_p,   // [batch, seq, seq, num_heads]
    int batch_size, int seq_len, int num_heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * seq_len * num_heads;
    if (idx >= total) return;

    int h = idx % num_heads;
    int j = (idx / num_heads) % seq_len;
    int i = ((idx / num_heads) / seq_len) % seq_len;
    int s = idx / (num_heads * seq_len * seq_len);

    float pi = p[idx];
    float dpi = d_p[idx];

    // Compute sum over j for this (s, i, h)
    float sum = 0.0f;
    for (int jj = 0; jj < seq_len; jj++) {
        sum += p[s * seq_len * seq_len * num_heads
                 + i * seq_len * num_heads + jj * num_heads + h]
             * d_p[s * seq_len * seq_len * num_heads
                   + i * seq_len * num_heads + jj * num_heads + h];
    }

    d_logit[idx] = pi * (dpi - sum);
}

// Kernel: dQ = dLogit @ K * scale
__global__ void attn_bwd_dq_kernel(
    float* d_q,
    const float* d_logit,     // [batch, seq, seq, num_heads]
    const float* k,           // [batch, seq, num_kv_heads, head_dim]
    float softmax_scale,
    int batch_size, int seq_len,
    int num_heads, int num_kv_heads, int head_dim) {
    int dim = threadIdx.x;
    if (dim >= head_dim) return;

    int sample = blockIdx.z / num_heads;
    int head = blockIdx.z % num_heads;
    int pos = blockIdx.y;
    int kv_head = head * num_kv_heads / num_heads;

    if (blockIdx.z >= batch_size * seq_len * num_heads) return;

    float dq = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        dq += d_logit[sample * seq_len * seq_len * num_heads
                      + pos * seq_len * num_heads + j * num_heads + head]
              * k[sample * seq_len * num_kv_heads * head_dim
                  + j * num_kv_heads * head_dim + kv_head * head_dim + dim];
    }
    d_q[blockIdx.z * head_dim + dim] = dq * softmax_scale;
}

// Kernel: dK = dLogit^T @ Q * scale
__global__ void attn_bwd_dk_kernel(
    float* d_k,
    const float* d_logit,     // [batch, seq, seq, num_heads]
    const float* q,           // [batch, seq, num_heads, head_dim]
    float softmax_scale,
    int batch_size, int seq_len,
    int num_heads, int num_kv_heads, int head_dim) {
    int dim = threadIdx.x;
    if (dim >= head_dim) return;

    int sample = blockIdx.z / num_kv_heads;
    int kv_head = blockIdx.z % num_kv_heads;
    int j = blockIdx.y;
    int head = kv_head * (num_heads / num_kv_heads);  // first query head

    if (blockIdx.z >= batch_size * seq_len * num_kv_heads) return;

    float dk = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        dk += d_logit[sample * seq_len * seq_len * num_heads
                      + i * seq_len * num_heads + j * num_heads + head]
              * q[sample * seq_len * num_heads * head_dim
                  + i * num_heads * head_dim + head * head_dim + dim];
    }
    d_k[blockIdx.z * head_dim + dim] = dk * softmax_scale;
}

// ============================================================
// RoPE backward: un-rotate d_output
// Forward: q[i] = q0*cos - q1*sin, q[hd/2] = q0*sin + q1*cos
// Backward: d_q0 = d_q[i]*cos + d_q[hd/2]*sin
//           d_q1 = -d_q[i]*sin + d_q[hd/2]*cos
// ============================================================

__global__ void rope_backward_kernel(
    float* d_input,   // [batch, seq, num_heads, head_dim], in=d_output, out=d_input (for q/k parts)
    const float* cos, // [seq, head_dim/2]
    const float* sin, // [seq, head_dim/2]
    int batch_size, int seq_len, int num_heads, int head_dim) {
    int sample = blockIdx.z;
    int pos = blockIdx.y;
    int head = blockIdx.x;
    int dim_pair = threadIdx.x;

    if (blockIdx.z >= batch_size) return;
    if (blockIdx.y >= seq_len) return;
    if (blockIdx.x >= num_heads) return;
    if (dim_pair >= head_dim / 2) return;

    float* dq = d_input + sample * seq_len * num_heads * head_dim
                          + pos * num_heads * head_dim
                          + head * head_dim;

    const float* c = cos + pos * (head_dim / 2) + dim_pair;
    const float* s = sin + pos * (head_dim / 2) + dim_pair;

    float cos_val = c[dim_pair];
    float sin_val = s[dim_pair];

    float dq0 = dq[dim_pair];
    float dq1 = dq[dim_pair + head_dim / 2];

    // Un-rotate the gradient
    dq[dim_pair]         = dq0 * cos_val - dq1 * sin_val;
    dq[dim_pair + head_dim / 2] = dq0 * sin_val + dq1 * cos_val;
}

void rope_backward(float* d_input,
                   const float* cos, const float* sin,
                   int batch_size, int seq_len,
                   int num_heads, int head_dim,
                   cudaStream_t stream) {
    dim3 block(32);
    dim3 grid(num_heads, seq_len, batch_size);

    rope_backward_kernel<<<grid, block, 0, stream>>>(
        d_input, cos, sin, batch_size, seq_len, num_heads, head_dim);

    MLLM_CUDA_LAUNCH_CHECK(cudaGetLastError());
}

// ============================================================
// Main attention backward entry point
// ============================================================

void attention_backward(float* d_q, float* d_k, float* d_v,
                        const float* d_output, const float* q,
                        const float* k, const float* v,
                        const float* softmax_output,  // saved P from forward
                        int batch_size, int seq_len,
                        int num_heads, int num_kv_heads,
                        int head_dim, float softmax_scale,
                        cudaStream_t stream) {
    int BLOCK = 256;

    // Allocate temp buffers on device (caller must free)
    // d_p: [batch, seq, seq, num_heads]
    // d_logit: [batch, seq, seq, num_heads]
    float *d_p, *d_logit;
    size_t dp_size = (size_t)batch_size * seq_len * seq_len * num_heads * sizeof(float);
    cudaMalloc(&d_p, dp_size);
    cudaMalloc(&d_logit, dp_size);

    // Step 1: dV = P^T @ d_output
    dim3 grid_dv(batch_size * seq_len, num_kv_heads, 1);
    attn_bwd_dv_kernel<<<grid_dv, BLOCK, 0, stream>>>(
        d_v, softmax_output, d_output,
        batch_size, seq_len, num_heads, num_kv_heads, head_dim);

    // Step 2: dP = d_output @ V^T
    dim3 grid_dp(seq_len, batch_size * seq_len, num_heads);
    attn_bwd_dp_kernel<<<grid_dp, 1, 0, stream>>>(
        d_p, d_output, v,
        batch_size, seq_len, num_heads, num_kv_heads, head_dim);

    // Step 3: softmax derivative — dLogit = P * (dP - sum(P*dP, dim=j))
    int total_sp = batch_size * seq_len * seq_len * num_heads;
    int grid_sp = (total_sp + BLOCK - 1) / BLOCK;
    attn_bwd_softmax_deriv_kernel<<<grid_sp, BLOCK, 0, stream>>>(
        d_logit, softmax_output, d_p,
        batch_size, seq_len, num_heads);

    // Step 4: dQ = dLogit @ K * scale
    dim3 grid_dq(num_heads, seq_len, batch_size);
    attn_bwd_dq_kernel<<<grid_dq, BLOCK, 0, stream>>>(
        d_q, d_logit, k, softmax_scale,
        batch_size, seq_len, num_heads, num_kv_heads, head_dim);

    // Step 5: dK = dLogit^T @ Q * scale
    dim3 grid_dk(num_kv_heads, seq_len, batch_size);
    attn_bwd_dk_kernel<<<grid_dk, BLOCK, 0, stream>>>(
        d_k, d_logit, q, softmax_scale,
        batch_size, seq_len, num_heads, num_kv_heads, head_dim);

    cudaFree(d_p);
    cudaFree(d_logit);

    MLLM_CUDA_LAUNCH_CHECK(cudaGetLastError());
}
