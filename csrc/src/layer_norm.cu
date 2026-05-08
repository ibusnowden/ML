// Copyright 2024 mmllm contributors
// LayerNorm implementation

#include "layer_norm.h"
#include "error.h"
#include "cuda_stream.h"
#include <cuda_runtime.h>

// Kernel declarations (defined in layer_norm_kernel.cu)
extern "C" {
    __global__ void layer_norm_kernel(float *out, const float *input,
                                      const float *weight, const float *bias,
                                      int hidden_size, int batch_size);
    __global__ void layer_norm_backward_kernel(float *d_input, float *d_weight,
                                                float *d_bias, const float *input,
                                                const float *weight, const float *grad_output,
                                                int hidden_size, int batch_size);
}

int layer_norm_forward(float *output, const float *input,
                       const float *weight, const float *bias,
                       int hidden_size, int batch_size, cudaStream_t stream) {
    const int BLOCK = 256;
    int grid = (batch_size + BLOCK - 1) / BLOCK;

    layer_norm_kernel<<<grid, BLOCK, 0, stream>>>(
        output, input, weight, bias, hidden_size, batch_size);

    return MLLM_CUDA_CHECK(cudaGetLastError());
}

int layer_norm_backward(float *d_input, float *d_weight, float *d_bias,
                        const float *input, const float *weight,
                        const float *grad_output, int hidden_size,
                        int batch_size, cudaStream_t stream) {
    const int BLOCK = 256;
    int grid = (batch_size + BLOCK - 1) / BLOCK;

    layer_norm_backward_kernel<<<grid, BLOCK, 0, stream>>>(
        d_input, d_weight, d_bias, input, weight, grad_output,
        hidden_size, batch_size);

    return MLLM_CUDA_CHECK(cudaGetLastError());
}
