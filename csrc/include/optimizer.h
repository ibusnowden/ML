// Copyright 2024 entropy contributors
// AdamW optimizer with FSDP-compatible gradient handling

#pragma once

#include "model.h"
#include "nccl_wrapper.h"
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Optimizer state per parameter group
typedef struct {
    float *momentum;    // First moment estimate
    float *velocity;    // Second moment estimate
} mllm_optimizer_state_t;

// AdamW optimizer
typedef struct {
    float lr;           // Learning rate
    float beta1;        // First moment decay
    float beta2;        // Second moment decay
    float eps;          // Epsilon for numerical stability
    float weight_decay; // L2 regularization
    int max_grad_norm;  // Gradient clipping threshold (0 = no clipping)

    mllm_model_t *model;
    mllm_comm_group_t *dp_comm;

    // Per-parameter optimizer states (simplified: flat arrays)
    mllm_optimizer_state_t *states;
    int num_param_groups;
    int64_t step;
} mllm_optimizer_t;

// Create optimizer
int mllm_optimizer_create(mllm_optimizer_t *opt, mllm_model_t *model,
                          mllm_comm_group_t *dp_comm,
                          float lr, float beta1, float beta2,
                          float eps, float weight_decay);

// Destroy optimizer
void mllm_optimizer_destroy(mllm_optimizer_t *opt);

// Step: apply gradients and update parameters
int mllm_optimizer_step(mllm_optimizer_t *opt, cudaStream_t stream);

// Zero gradients
int mllm_optimizer_zero_grad(mllm_optimizer_t *opt, cudaStream_t stream);

// Gradient clipping
int mllm_optimizer_clip_grads(mllm_optimizer_t *opt, float max_norm,
                               cudaStream_t stream);

#ifdef __cplusplus
}
#endif
