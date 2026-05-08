// Copyright 2024 mmllm contributors
// MoE (Mixture of Experts) kernel implementation

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

template <typename T>
__global__ void moe_router_kernel(int* selected_experts, 
                                   const float* input,
                                   const float* router_weights,
                                   int batch_size, int seq_len, int hidden_size,
                                   int num_experts, int top_k) {
    int tid = threadIdx.x;
    
    for (int sample = blockIdx.x; sample < batch_size; sample += gridDim.x) {
        for (int pos = blockIdx.y; pos < seq_len; pos += blockDim.y) {
            int idx = sample * seq_len + pos;
            
            // Compute router scores
            float scores[64]; // Max experts
            for (int e = tid; e < num_experts; e += 32) {
                float score = 0.0f;
                for (int j = 0; j < hidden_size; j += 32) {
                    score += static_cast<float>(input[idx * hidden_size + j]) *
                             static_cast<float>(router_weights[j * num_experts + e]);
                }
                scores[e] = score;
            }
            
            // Find top-k experts (simple selection for now)
            int selected[8];
            bool used[64] = {false};
            for (int k = 0; k < top_k && threadIdx.x == 0; k++) {
                float max_val = -INFINITY;
                int max_idx = -1;
                for (int e = 0; e < num_experts; e++) {
                    if (!used[e] && scores[e] > max_val) {
                        max_val = scores[e];
                        max_idx = e;
                    }
                }
                if (max_idx != -1) {
                    used[max_idx] = true;
                    selected[k] = max_idx;
                }
            }
            
            // Write selected experts
            for (int k = 0; k < top_k; k++) {
                if (tid < top_k) {
                    selected_experts[idx * top_k + tid] = selected[tid];
                }
            }
        }
    }
}

void moe_router_forward(int* selected_experts, const float* input,
                        const float* router_weights,
                        int batch_size, int seq_len, int hidden_size,
                        int num_experts, int top_k, cudaStream_t stream) {
    dim3 block(32);
    dim3 grid(batch_size, seq_len);
    
    moe_router_kernel<float><<<grid, block, 0, stream>>>(
        selected_experts, input, router_weights,
        batch_size, seq_len, hidden_size, num_experts, top_k);
    
    MLLM_CUDA_LAUNCH_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void moe_gating_kernel(float* gating_weights,
                                    const float* input,
                                    const float* router_weights,
                                    int batch_size, int seq_len, int hidden_size,
                                    int num_experts, int top_k) {
    int tid = threadIdx.x;
    
    for (int sample = blockIdx.x; sample < batch_size; sample += gridDim.x) {
        for (int pos = blockIdx.y; pos < seq_len; pos += blockDim.y) {
            int idx = sample * seq_len + pos;
            
            float scores[64];
            for (int e = tid; e < num_experts; e += 32) {
                float score = 0.0f;
                for (int j = 0; j < hidden_size; j += 32) {
                    score += static_cast<float>(input[idx * hidden_size + j]) *
                             static_cast<float>(router_weights[j * num_experts + e]);
                }
                scores[e] = score;
            }
            
            // Softmax over top-k
            float max_score = -INFINITY;
            for (int e = tid; e < num_experts; e += 32) {
                if (scores[e] > max_score) max_score = scores[e];
            }
            for (int offset = 16; offset > 0; offset /= 2) {
                max_score += __shfl_down_sync(0xFFFFFFFF, max_score, offset);
            }
            
            float sum_exp = 0.0f;
            for (int e = tid; e < num_experts; e += 32) {
                scores[e] = expf(scores[e] - max_score);
                sum_exp += scores[e];
            }
            for (int offset = 16; offset > 0; offset /= 2) {
                sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
            }
            
            for (int e = tid; e < top_k; e += 32) {
                if (e < top_k) {
                    gating_weights[idx * top_k + e] = 
                        scores[e] / (sum_exp + 1e-8f);
                }
            }
        }
    }
}

void moe_gating_forward(float* gating_weights, const float* input,
                        const float* router_weights,
                        int batch_size, int seq_len, int hidden_size,
                        int num_experts, int top_k, cudaStream_t stream) {
    dim3 block(32);
    dim3 grid(batch_size, seq_len);
    
    moe_gating_kernel<float><<<grid, block, 0, stream>>>(
        gating_weights, input, router_weights,
        batch_size, seq_len, hidden_size, num_experts, top_k);
    
    MLLM_CUDA_LAUNCH_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void moe_forward_kernel(float* output, const float* input,
                                    const float* expert_weights,
                                    const int* selected_experts,
                                    const float* gating_weights,
                                    int batch_size, int seq_len, int hidden_size,
                                    int intermediate_size, int num_experts, int top_k) {
    int tid = threadIdx.x;
    
    for (int sample = blockIdx.x; sample < batch_size; sample += gridDim.x) {
        for (int pos = blockIdx.y; pos < seq_len; pos += blockDim.y) {
            int idx = sample * seq_len + pos;
            float out_val = 0.0f;
            
            for (int k = 0; k < top_k; k++) {
                int expert = selected_experts[idx * top_k + k];
                float gate = gating_weights[idx * top_k + k];
                
                const float* expert_w = expert_weights + 
                    expert * hidden_size * intermediate_size;
                
                for (int j = tid; j < hidden_size; j += 32) {
                    float val = static_cast<float>(input[idx * hidden_size + j]);
                    out_val += val * expert_w[j * intermediate_size + tid];
                }
                out_val *= gate;
            }
            
            output[idx * hidden_size + tid] = static_cast<T>(out_val);
        }
    }
}

void moe_forward(float* output, const float* input,
                const float* expert_weights, const int* selected_experts,
                const float* gating_weights,
                int batch_size, int seq_len, int hidden_size,
                int intermediate_size, int num_experts, int top_k,
                cudaStream_t stream) {
    dim3 block(32);
    dim3 grid(batch_size, seq_len);
    
    moe_forward_kernel<float><<<grid, block, 0, stream>>>(
        output, input, expert_weights, selected_experts, gating_weights,
        batch_size, seq_len, hidden_size, intermediate_size, num_experts, top_k);
    
    MLLM_CUDA_LAUNCH_CHECK(cudaGetLastError());
}
