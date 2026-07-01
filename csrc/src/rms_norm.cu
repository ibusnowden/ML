// Copyright 2024 entropy contributors
// RMSNorm implementation (used in modern transformers instead of LayerNorm)

#include "error.h"
#include <cuda_runtime.h>

__global__ void rms_norm_forward_kernel(float *output, const float *input,
                                         const float *weight, int size,
                                         float eps, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float *x = input + idx * size;
    float *o = output + idx * size;

    // Compute RMS
    float rsum = 0.0f;
    for (int j = 0; j < size; ++j) {
        rsum += x[j] * x[j];
    }
    float rms = rsqrtf(rsum / size + eps);

    // Normalize and scale
    for (int j = 0; j < size; ++j) {
        o[j] = (x[j] * rms) * weight[j];
    }
}

__global__ void rms_norm_backward_kernel(float *d_input, float *d_weight,
                                          const float *input, const float *weight,
                                          const float *grad_output, int size,
                                          float eps, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float *x = input + idx * size;
    const float *g = grad_output + idx * size;
    float *di = d_input + idx * size;

    // Compute RMS
    float rsum = 0.0f;
    for (int j = 0; j < size; ++j) {
        rsum += x[j] * x[j];
    }
    float rms = rsqrtf(rsum / size + eps);

    // Compute weighted sum for reduction
    float sum_weighted_g = 0.0f;
    for (int j = 0; j < size; ++j) {
        sum_weighted_g += g[j] * (x[j] * rms) * weight[j];
    }
    sum_weighted_g /= size;

    // Compute d_input
    for (int j = 0; j < size; ++j) {
        float norm_x = x[j] * rms;
        di[j] = (g[j] * weight[j] - norm_x * sum_weighted_g) * rms;
        atomicAdd(&d_weight[j], g[j] * norm_x);
    }
}

int rms_norm_forward(float *output, const float *input,
                     const float *weight, int hidden_size, int rows,
                     float eps, cudaStream_t stream) {
    const int BLOCK = 256;
    int grid = (rows + BLOCK - 1) / BLOCK;

    rms_norm_forward_kernel<<<grid, BLOCK, 0, stream>>>(
        output, input, weight, hidden_size, eps, rows);

    return MLLM_CUDA_CHECK(cudaGetLastError());
}

int rms_norm_backward(float *d_input, float *d_weight,
                      const float *input, const float *weight,
                      const float *grad_output, int hidden_size, int rows,
                      float eps, cudaStream_t stream) {
    const int BLOCK = 256;
    int grid = (rows + BLOCK - 1) / BLOCK;

    rms_norm_backward_kernel<<<grid, BLOCK, 0, stream>>>(
        d_input, d_weight, input, weight, grad_output, hidden_size, eps, rows);

    return MLLM_CUDA_CHECK(cudaGetLastError());
}
