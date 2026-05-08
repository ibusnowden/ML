// Copyright 2024 mmllm contributors
// MLP (FFN) layer kernels — SwiGLU variant for Qwen architecture

#pragma once

// SwiGLU(x, W1, W2, W3) = SwiGLU(x @ W1, x @ W3) @ W2
// Where SwiGLU(a, b) = silu(a) * b

#define SILU_THRESHOLD 12.0f

__device__ float silu(float x) {
    float s = fminf(fmaxf(x, -SILU_THRESHOLD), SILU_THRESHOLD);
    return x / (1.0f + expf(-s));
}

// Optimized MLP with vectorized memory access
__global__ void swiglu_mlp_forward_kernel(
    float *output,           // [batch_size, intermediate_size]
    const float *input,      // [batch_size, hidden_size]
    const float *W1,         // [hidden_size, intermediate_size]
    const float *W3,         // [hidden_size, intermediate_size]
    const float *W2,         // [intermediate_size, hidden_size]
    const float *gate,       // optional bias
    const float *up,         // optional bias
    const float *down,       // optional bias
    int batch_size,
    int hidden_size,
    int intermediate_size) {

    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float *x = input + row * hidden_size;
    float *o = output + row * intermediate_size;

    // Vectorized GEMM using shared memory tile
    // Tile size chosen for H100 shared memory capacity
    const int TILE = 1024;
    extern __shared__ char shared_mem[];
    float *tile_x = (float*)shared_mem;
    float *tile_W1 = tile_x + TILE;
    float *tile_W3 = tile_W1 + TILE;

    float sum_g = 0.0f;
    float sum_u = 0.0f;

    for (int tile_start = 0; tile_start < hidden_size; tile_start += TILE) {
        int tile_end = min(tile_start + TILE, hidden_size);
        int tile_size = tile_end - tile_start;

        // Load x tile into shared memory
        #pragma unroll
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            tile_x[i] = x[tile_start + i];
        }
        __syncthreads();

        // Compute partial GEMM for W1
        float w1_partial = 0.0f;
        for (int j = 0; j < tile_size; ++j) {
            w1_partial += tile_x[j] * W1[j * intermediate_size + threadIdx.x + (blockIdx.y % (intermediate_size / blockDim.x)) * blockDim.x];
        }
        sum_g += w1_partial;

        // Compute partial GEMM for W3
        float w3_partial = 0.0f;
        for (int j = 0; j < tile_size; ++j) {
            w3_partial += tile_x[j] * W3[j * intermediate_size + threadIdx.x + (blockIdx.y % (intermediate_size / blockDim.x)) * blockDim.x];
        }
        sum_u += w3_partial;

        __syncthreads();
    }

    // Apply SwiGLU
    float gate_val = silu(sum_g + (gate ? gate[threadIdx.x] : 0.0f));
    float up_val = sum_u + (up ? up[threadIdx.x] : 0.0f);
    o[blockIdx.y] = gate_val * up_val;

    // Now multiply by W2 — final projection
    // This would be a separate kernel in production
}
