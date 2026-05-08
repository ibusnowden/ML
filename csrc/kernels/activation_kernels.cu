// Copyright 2024 mmllm contributors
// LayerNorm CUDA kernel

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

#define MLLM_CUDA_LAUNCH_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA launch error: %s\n", cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

template <typename T>
__global__ void layer_norm_kernel(T* output, const T* input, const T* weight, 
                                   const T* bias, const float epsilon,
                                   int hidden_size, int batch_size) {
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    int tid = threadIdx.x;
    
    for (int sample = blockIdx.x; sample < batch_size; sample += gridDim.x) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        for (int j = tid; j < hidden_size; j += 32) {
            float val = static_cast<float>(input[sample * hidden_size + j]);
            sum += val;
            sum_sq += val * val;
        }
        
        // Warp reduction for sum
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        
        // Warp reduction for sum_sq
        float sum_sq_warp = sum_sq;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_sq_warp += __shfl_down_sync(0xFFFFFFFF, sum_sq_warp, offset);
        }
        
        if (warp == 0 && lane == 0) {
            float mean = sum / static_cast<float>(hidden_size);
            float variance = sum_sq_warp / static_cast<float>(hidden_size) - mean * mean;
            
            // Compute normalized values and write
            for (int j = tid; j < hidden_size; j += 32) {
                float val = static_cast<float>(input[sample * hidden_size + j]);
                float normalized = (val - mean) / sqrtf(variance + epsilon);
                float scaled = normalized * static_cast<float>(weight[j]) + 
                              static_cast<float>(bias[j]);
                output[sample * hidden_size + j] = static_cast<T>(scaled);
            }
        }
    }
}

void layer_norm_forward(float* output, const float* input, const float* weight,
                        const float* bias, int hidden_size, int batch_size,
                        float epsilon, cudaStream_t stream) {
    const int MAX_BLOCK_DIM = 256;
    const int num_blocks = 256; // Tune based on batch size
    
    layer_norm_kernel<float><<<num_blocks, MAX_BLOCK_DIM, 0, stream>>>(
        output, input, weight, bias, epsilon, hidden_size, batch_size);
}

template <typename T>
__global__ void layer_norm_backward_kernel(T* d_input, const T* d_output, 
                                            const T* input, const T* weight,
                                            const float epsilon,
                                            int hidden_size, int batch_size) {
    int tid = threadIdx.x;
    
    for (int sample = blockIdx.x; sample < batch_size; sample += gridDim.x) {
        float sum_dy = 0.0f;
        float sum_dy_xhat = 0.0f;
        
        for (int j = tid; j < hidden_size; j += 32) {
            float dy = static_cast<float>(d_output[sample * hidden_size + j]);
            float x_hat = (static_cast<float>(input[sample * hidden_size + j]) - 
                         static_cast<float>(weight[j])) / 
                         sqrtf(static_cast<float>(bias[j]) + epsilon);
            sum_dy += dy;
            sum_dy_xhat += dy * x_hat;
        }
        
        // Warp reductions
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_dy += __shfl_down_sync(0xFFFFFFFF, sum_dy, offset);
            sum_dy_xhat += __shfl_down_sync(0xFFFFFFFF, sum_dy_xhat, offset);
        }
        
        if (threadIdx.x == 0) {
            float mean_dy = sum_dy / static_cast<float>(hidden_size);
            float mean_dy_xhat = sum_dy_xhat / static_cast<float>(hidden_size);
            
            for (int j = tid; j < hidden_size; j += 32) {
                float dy = static_cast<float>(d_output[sample * hidden_size + j]);
                float x = static_cast<float>(input[sample * hidden_size + j]);
                float inv_var = 1.0f / sqrtf(static_cast<float>(bias[j]) + epsilon);
                
                float dx = (dy - mean_dy - 
                           static_cast<float>(weight[j]) * mean_dy_xhat) * 
                          inv_var;
                d_input[sample * hidden_size + j] = static_cast<T>(dx);
            }
        }
    }
}

void layer_norm_backward(float* d_input, const float* d_output, const float* input,
                         const float* weight, const float* variance,
                         int hidden_size, int batch_size, float epsilon,
                         cudaStream_t stream) {
    const int MAX_BLOCK_DIM = 256;
    const int num_blocks = 256;
    
    layer_norm_backward_kernel<float><<<num_blocks, MAX_BLOCK_DIM, 0, stream>>>(
        d_input, d_output, input, weight, epsilon, hidden_size, batch_size);
}
