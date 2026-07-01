// Copyright 2024 entropy contributors
// Fully Sharded Data Parallel (FSDP) manager
// Shards model parameters, gradients, and optimizer states across data parallel ranks

#pragma once

#include "model.h"
#include "optimizer.h"
#include "nccl_wrapper.h"
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// FSDP state
typedef struct {
    mllm_model_t *model;
    mllm_optimizer_t *optimizer;
    mllm_comm_group_t *dp_comm;  // Data parallel group

    // Shard info
    int dp_rank;
    int dp_size;
    size_t local_param_bytes;
    size_t total_param_bytes;

    // Communication buffers for gradient all-reduce
    float *grad_allreduce_buf;

    // Training state
    int64_t step;
    int64_t tokens_seen;
    float train_loss;
} mllm_fsdp_t;

typedef enum {
    MLLM_IMAGE_BATCH_NONE = 0,
    MLLM_IMAGE_BATCH_HOST_PATCHES_F32 = 1,
    MLLM_IMAGE_BATCH_DEVICE_PATCHES_F32 = 2,
} mllm_image_batch_location_t;

// Optional image batch passed to FSDP. The buffer is patch-major float32:
// [batch_size, num_images, values_per_image], where values_per_image is
// 3 * patch_size * patch_size for the legacy path and
// num_patches * 3 * patch_size * patch_size for the ViT path.
typedef struct {
    const float *data;
    int batch_size;
    int num_images;
    int values_per_image;
    mllm_image_batch_location_t location;
} mllm_image_batch_t;

// Create FSDP manager
int mllm_fsdp_create(mllm_fsdp_t *fsdp, mllm_model_t *model,
                     mllm_optimizer_t *optimizer,
                     mllm_comm_group_t *dp_comm);

// Destroy FSDP manager
void mllm_fsdp_destroy(mllm_fsdp_t *fsdp);

// Single training step: forward + backward + gradient all-reduce + optimizer step
int mllm_fsdp_step(mllm_fsdp_t *fsdp,
                   const int *input_ids,
                   const int *position_ids,
                   const float *image_embeds,
                   const int *labels,
                   int batch_size,
                   int seq_len,
                   cudaStream_t stream);

int mllm_fsdp_step_with_image_batch(mllm_fsdp_t *fsdp,
                                    const int *input_ids,
                                    const int *position_ids,
                                    const mllm_image_batch_t *image_batch,
                                    const int *labels,
                                    int batch_size,
                                    int seq_len,
                                    cudaStream_t stream);

// Synchronize gradients across data parallel ranks via all-reduce
int mllm_fsdp_all_reduce_grads(mllm_fsdp_t *fsdp, cudaStream_t stream);

// Gather full model weights for checkpointing (broadcast from rank 0)
int mllm_fsdp_gather_weights(mllm_fsdp_t *fsdp, void *full_weights,
                              cudaStream_t stream);

// Scatter sharded weights from full weights (for loading from checkpoint)
int mllm_fsdp_scatter_weights(mllm_fsdp_t *fsdp, const void *full_weights,
                               cudaStream_t stream);

#ifdef __cplusplus
}
#endif
