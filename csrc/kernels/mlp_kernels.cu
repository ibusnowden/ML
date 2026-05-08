// Copyright 2024 mmllm contributors
// MLP (Gated Linear Unit) kernel implementation

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

#define MLLM_CUDA_LAUNCH_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

template <typename T>
__global__ void mlp_forward_kernel(T* output, const T* input,
                                    const T* w_up, const T* w_gate, const T* w_down,
                                    const T* b_up, const T* b_gate, const T* b_down,
                                    int batch_size, int seq_len, int hidden_size,
                                    int intermediate_size) {
    // Each thread block handles one (batch, seq) position
    int tid = threadIdx.x;
    int lane = tid % 32;
    
    for (int sample = blockIdx.x; sample < batch_size; sample += gridDim.x) {
        for (int pos = blockIdx.y; pos < seq_len; pos += gridBlockSize.y) {
            const int idx = sample * seq_len + pos;
            
            // Compute up projection: up = gelu(x @ w_up + b_up)
            float up_val = 0.0f;
            for (int j = tid; j < hidden_size; j += 32) {
                up_val += static_cast<float>(input[idx * hidden_size + j]) * 
                         static_cast<float>(w_up[j * intermediate_size + tid]);
            }
            up_val += static_cast<float>(b_up[tid]);
            
            // Compute gate projection: gate = silu(x @ w_gate + b_gate)
            float gate_val = 0.0f;
            for (int j = tid; j < hidden_size; j += 32) {
                gate_val += static_cast<float>(input[idx * hidden_size + j]) * 
                           static_cast<float>(w_gate[j * intermediate_size + tid]);
            }
            gate_val += static_cast<float>(b_gate[tid]);
            
            // Apply SiLU activation
            gate_val = gate_val / (1.0f + expf(-gate_val));
            
            // Multiply up * gate
            float activated = up_val * gate_val;
            
            // Down projection
            float down_val = activated * static_cast<float>(w_down[tid * intermediate_size + tid]);
            
            output[idx * hidden_size + tid] = static_cast<T>(down_val);
        }
    }
}

void mlp_forward(float* output, const float* input,
                const float* w_up, const float* w_gate, const float* w_down,
                const float* b_up, const float* b_gate, const float* b_down,
                int batch_size, int seq_len, int hidden_size, int intermediate_size,
                cudaStream_t stream) {
    dim3 block(32);
    dim3 grid(batch_size, seq_len);
    
    mlp_forward_kernel<float><<<grid, block, 0, stream>>>(
        output, input, w_up, w_gate, w_down, b_up, b_gate, b_down,
        batch_size, seq_len, hidden_size, intermediate_size);
    
    MLLM_CUDA_LAUNCH_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void mlp_backward_kernel(float* d_input, const float* d_output,
                                     const float* input, const float* w_up, 
                                     const float* w_gate, const float* w_down,
                                     int batch_size, int seq_len, int hidden_size,
                                     int intermediate_size) {
    int tid = threadIdx.x;
    
    for (int sample = blockIdx.x; sample < batch_size; sample += gridDim.x) {
        for (int pos = blockIdx.y; pos < seq_len; pos += blockDim.y) {
            int idx = sample * seq_len + pos;
            
            // Forward values needed for backward
            float up_val = 0.0f;
            for (int j = tid; j < hidden_size; j += 32) {
                up_val += static_cast<float>(input[idx * hidden_size + j]) * 
                         static_cast<float>(w_up[j * intermediate_size + tid]);
            }
            up_val += static_cast<float>(b_up[tid]);
            
            float gate_val = 0.0f;
            for (int j = tid; j < hidden_size; j += 32) {
                gate_val += static_cast<float>(input[idx * hidden_size + j]) * 
                           static_cast<float>(w_gate[j * intermediate_size + tid]);
            }
            gate_val += static_cast<float>(b_gate[tid]);
            float gate_act = gate_val / (1.0f + expf(-gate_val));
            float gate_deriv = gate_act * (1.0f - gate_act);
            
            float activated = up_val * gate_act;
            float d_down = static_cast<float>(d_output[idx * hidden_size + tid]);
            
            // Gradient through down projection
            float d_activated = d_down * static_cast<float>(w_down[tid * intermediate_size + tid]);
            
            // Gradient through up * gate
            float d_up = d_activated * gate_act;
            float d_gate = d_activated * up_val;
            
            // Accumulate gradients for input
            for (int j = 0; j < hidden_size; j += 32) {
                float x = static_cast<float>(input[idx * hidden_size + j]);
                atomicAdd(&d_input[idx * hidden_size + j],
                         d_up * static_cast<float>(w_up[j * intermediate_size + tid]) +
                         d_gate * gate_deriv * static_cast<float>(w_gate[j * intermediate_size + tid]));
            }
        }
    }
}

void mlp_backward(float* d_input, const float* d_output, const float* input,
                 const float* w_up, const float* w_gate, const float* w_down,
                 const float* b_up, const float* b_gate,
                 int batch_size, int seq_len, int hidden_size, int intermediate_size,
                 cudaStream_t stream) {
    dim3 block(32);
    dim3 grid(batch_size, seq_len);
    
    mlp_backward_kernel<float><<<grid, block, 0, stream>>>(
        d_input, d_output, input, w_up, w_gate, w_down,
        batch_size, seq_len, hidden_size, intermediate_size);
    
    MLLM_CUDA_LAUNCH_CHECK(cudaGetLastError());
}
