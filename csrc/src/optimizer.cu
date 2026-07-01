// Copyright 2024 entropy contributors
// AdamW optimizer implementation

#include "optimizer.h"
#include "error.h"
#include "nccl_wrapper.h"
#include "logging.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Kernel: AdamW update step
__global__ void adamw_step_kernel(float *params, float *momentum, float *velocity,
                                   float *grads, size_t total_elements,
                                   float lr, float beta1, float beta2, float eps,
                                   float weight_decay, float t) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float g = grads[idx];
    float m = momentum[idx];
    float v = velocity[idx];

    // Update biased first moment estimate
    m = beta1 * m + (1.0f - beta1) * g;
    momentum[idx] = m;

    // Update biased second moment estimate
    v = beta2 * v + (1.0f - beta2) * g * g;
    velocity[idx] = v;

    // Bias correction
    float m_hat = m / (1.0f - powf(beta1, t));
    float v_hat = v / (1.0f - powf(beta2, t));

    // Update parameters with weight decay
    params[idx] = params[idx] * (1.0f - lr * weight_decay) -
                  lr * m_hat / (sqrtf(v_hat) + eps);
}

int mllm_optimizer_create(mllm_optimizer_t *opt, mllm_model_t *model,
                          mllm_comm_group_t *dp_comm,
                          float lr, float beta1, float beta2,
                          float eps, float weight_decay) {
    memset(opt, 0, sizeof(*opt));
    opt->model = model;
    opt->dp_comm = dp_comm;
    opt->lr = lr;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->eps = eps;
    opt->weight_decay = weight_decay;
    opt->num_param_groups = 2;
    opt->step = 0;

    // Parameter allocations are not contiguous. Keep state for the two
    // gradients currently produced by the stable backward path.
    size_t tok_params = (size_t)model->config.vocab_size * model->config.hidden_size;
    size_t out_params = (size_t)model->config.hidden_size * model->config.vocab_size;
    opt->states = (mllm_optimizer_state_t *)calloc(opt->num_param_groups, sizeof(mllm_optimizer_state_t));
    if (!opt->states) return MLLM_ERR_ALLOC;
    MLLM_CUDA_CHECK(cudaMalloc(&opt->states[0].momentum, tok_params * sizeof(float)));
    MLLM_CUDA_CHECK(cudaMalloc(&opt->states[0].velocity, tok_params * sizeof(float)));
    MLLM_CUDA_CHECK(cudaMalloc(&opt->states[1].momentum, out_params * sizeof(float)));
    MLLM_CUDA_CHECK(cudaMalloc(&opt->states[1].velocity, out_params * sizeof(float)));
    MLLM_CUDA_CHECK(cudaMemset(opt->states[0].momentum, 0, tok_params * sizeof(float)));
    MLLM_CUDA_CHECK(cudaMemset(opt->states[0].velocity, 0, tok_params * sizeof(float)));
    MLLM_CUDA_CHECK(cudaMemset(opt->states[1].momentum, 0, out_params * sizeof(float)));
    MLLM_CUDA_CHECK(cudaMemset(opt->states[1].velocity, 0, out_params * sizeof(float)));

    MLLM_LOG_INFO("Optimizer created: lr=%.6f, beta1=%.3f, beta2=%.3f, wd=%.4f",
                  lr, beta1, beta2, weight_decay);
    return MLLM_OK;
}

void mllm_optimizer_destroy(mllm_optimizer_t *opt) {
    if (!opt) return;
    if (opt->states) {
        for (int i = 0; i < opt->num_param_groups; i++) {
            cudaFree(opt->states[i].momentum);
            cudaFree(opt->states[i].velocity);
        }
        free(opt->states);
    }
    memset(opt, 0, sizeof(*opt));
}

int mllm_optimizer_step(mllm_optimizer_t *opt, cudaStream_t stream) {
    mllm_model_t *model = opt->model;
    size_t tok_params = (size_t)model->config.vocab_size * model->config.hidden_size;
    size_t out_params = (size_t)model->config.hidden_size * model->config.vocab_size;
    opt->step++;

    int BLOCK = 256;
    int tok_grid = (tok_params + BLOCK - 1) / BLOCK;
    int out_grid = (out_params + BLOCK - 1) / BLOCK;

    adamw_step_kernel<<<tok_grid, BLOCK, 0, stream>>>(
        model->tok_embeddings,
        opt->states[0].momentum,
        opt->states[0].velocity,
        model->grad_tok_embeddings,
        tok_params,
        opt->lr, opt->beta1, opt->beta2, opt->eps,
        opt->weight_decay,
        (float)opt->step);

    adamw_step_kernel<<<out_grid, BLOCK, 0, stream>>>(
        model->output_weight,
        opt->states[1].momentum,
        opt->states[1].velocity,
        model->grad_output_weight,
        out_params,
        opt->lr, opt->beta1, opt->beta2, opt->eps,
        opt->weight_decay,
        (float)opt->step);

    return MLLM_CUDA_CHECK(cudaGetLastError());
}

int mllm_optimizer_zero_grad(mllm_optimizer_t *opt, cudaStream_t stream) {
    mllm_model_t *model = opt->model;
    MLLM_CUDA_CHECK(cudaMemsetAsync(model->grad_tok_embeddings, 0,
                    (size_t)model->config.vocab_size * model->config.hidden_size * sizeof(float), stream));
    MLLM_CUDA_CHECK(cudaMemsetAsync(model->grad_output_weight, 0,
                    (size_t)model->config.hidden_size * model->config.vocab_size * sizeof(float), stream));
    return MLLM_OK;
}

int mllm_optimizer_clip_grads(mllm_optimizer_t *opt, float max_norm,
                               cudaStream_t stream) {
    (void)opt;
    (void)max_norm;
    (void)stream;
    return MLLM_OK;
}
