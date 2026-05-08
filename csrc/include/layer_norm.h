// Copyright 2024 mmllm contributors
// LayerNorm interface

#pragma once

#include <cuda_runtime.h>

// Forward: layer_norm(x) = (x - mean) / sqrt(var + eps) * weight + bias
// x: [batch_size, hidden_size]
// output: [batch_size, hidden_size]
int layer_norm_forward(float *output, const float *input,
                       const float *weight, const float *bias,
                       int hidden_size, int batch_size, cudaStream_t stream);

// Backward: compute gradients
int layer_norm_backward(float *d_input, float *d_weight, float *d_bias,
                        const float *input, const float *weight,
                        const float *grad_output, int hidden_size,
                        int batch_size, cudaStream_t stream);
