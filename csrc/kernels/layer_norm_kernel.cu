// Copyright 2024 entropy contributors
// CUDA kernels for LayerNorm

#include <cuda_runtime.h>
#include <stdint.h>

// Forward: LayerNorm over the last dimension
// input:  [batch_size, hidden_size]
// weight: [hidden_size]  (learnable scale)
// bias:   [hidden_size]  (learnable shift)
// out:    [batch_size, hidden_size]
extern "C" __global__ void layer_norm_kernel(float *out, const float *input,
                                  const float *weight, const float *bias,
                                  int hidden_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float *x = input + idx * hidden_size;
    float *o = out + idx * hidden_size;

    // Compute mean
    float sum = 0.0f;
    for (int j = 0; j < hidden_size; ++j) {
        sum += x[j];
    }
    float mean = sum / hidden_size;

    // Compute variance (population variance)
    float var = 0.0f;
    for (int j = 0; j < hidden_size; ++j) {
        float d = x[j] - mean;
        var += d * d;
    }
    float inv_std = rsqrtf(var / hidden_size + 1e-5f);

    // Normalize and apply scale/shift
    for (int j = 0; j < hidden_size; ++j) {
        float n = (x[j] - mean) * inv_std;
        o[j] = n * weight[j] + bias[j];
    }
}

// Backward: compute d_input, d_weight, d_bias
// grad_output: [batch_size, hidden_size]  (dL/dout)
// input:       [batch_size, hidden_size]  (forward input, needed for backward)
// weight:      [hidden_size]              (used for scale)
// inv_std:     [batch_size]               (precomputed rsqrt(var/hidden + eps))
// d_input:     [batch_size, hidden_size]
// d_weight:    [hidden_size]              (caller must zero before call, atomicAdd)
// d_bias:      [hidden_size]              (caller must zero before call, atomicAdd)
extern "C" __global__ void layer_norm_backward_kernel(float *d_input, float *d_weight,
                                            float *d_bias, const float *input,
                                            const float *weight, const float *grad_output,
                                            int hidden_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float *x = input + idx * hidden_size;
    const float *g = grad_output + idx * hidden_size;
    float *di = d_input + idx * hidden_size;
    float mean = 0.0f;
    for (int j = 0; j < hidden_size; ++j) mean += x[j];
    mean /= hidden_size;
    float var = 0.0f;
    for (int j = 0; j < hidden_size; ++j) {
        float d = x[j] - mean;
        var += d * d;
    }
    float inv_s = rsqrtf(var / hidden_size + 1e-5f);

    float sum_dy = 0.0f;
    float sum_dy_xhat = 0.0f;
    for (int j = 0; j < hidden_size; ++j) {
        float x_hat = (x[j] - mean) * inv_s;
        sum_dy += g[j] * weight[j];
        sum_dy_xhat += g[j] * weight[j] * x_hat;
        atomicAdd(&d_weight[j], g[j] * x_hat);
        atomicAdd(&d_bias[j], g[j]);
    }
    sum_dy /= hidden_size;
    sum_dy_xhat /= hidden_size;

    for (int j = 0; j < hidden_size; ++j) {
        float x_hat = (x[j] - mean) * inv_s;
        di[j] = (g[j] * weight[j] - sum_dy - x_hat * sum_dy_xhat) * inv_s;
    }
}
