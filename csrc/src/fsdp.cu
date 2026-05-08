// Copyright 2024 mmllm contributors
// FSDP (Fully Sharded Data Parallel) implementation

#include "fsdp.h"
#include "error.h"
#include "nccl_wrapper.h"
#include "comm.h"
#include "logging.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

#define FSDP_CUDA_GOTO(call) \
    do { \
        cudaError_t err__ = (call); \
        if (err__ != cudaSuccess) { \
            MLLM_LOG_ERROR("CUDA error %s at %s:%d", \
                           cudaGetErrorString(err__), __FILE__, __LINE__); \
            rc = MLLM_ERR_CUDA; \
            goto cleanup; \
        } \
    } while (0)

static int fsdp_image_values_per_image(const mllm_model_config_t *config) {
    int patch_size = config->vision_patch_size ? config->vision_patch_size : 14;
    int image_size = config->vision_image_size ? config->vision_image_size : config->image_size;
    if (image_size <= 0) image_size = 336;
    if (patch_size <= 0 || image_size % patch_size != 0) return -1;

    int patch_pixels = 3 * patch_size * patch_size;
    if (config->vision_num_layers <= 0) return patch_pixels;

    int patches_per_side = image_size / patch_size;
    return patches_per_side * patches_per_side * patch_pixels;
}

int mllm_fsdp_create(mllm_fsdp_t *fsdp, mllm_model_t *model,
                     mllm_optimizer_t *optimizer,
                     mllm_comm_group_t *dp_comm) {
    memset(fsdp, 0, sizeof(*fsdp));
    fsdp->model = model;
    fsdp->optimizer = optimizer;
    fsdp->dp_comm = dp_comm;
    fsdp->dp_rank = dp_comm->rank;
    fsdp->dp_size = dp_comm->world_size;
    fsdp->total_param_bytes = model->total_bytes;
    fsdp->local_param_bytes = model->total_bytes / dp_comm->world_size;
    fsdp->step = 0;
    fsdp->tokens_seen = 0;
    fsdp->train_loss = 0.0f;

    // Parameters are allocated separately; gradients are reduced in-place per buffer.
    fsdp->grad_allreduce_buf = NULL;

    MLLM_LOG_INFO("FSDP created: dp_rank=%d dp_size=%d total_params=%zu",
                  fsdp->dp_rank, fsdp->dp_size, model->total_params);
    return MLLM_OK;
}

void mllm_fsdp_destroy(mllm_fsdp_t *fsdp) {
    if (!fsdp) return;
    if (fsdp->grad_allreduce_buf) {
        cudaFree(fsdp->grad_allreduce_buf);
    }
    memset(fsdp, 0, sizeof(*fsdp));
}

__global__ void scale_kernel(float *data, size_t n, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= scale;
}

int mllm_fsdp_all_reduce_grads(mllm_fsdp_t *fsdp, cudaStream_t stream) {
    if (fsdp->dp_size <= 1) return MLLM_OK;

    mllm_model_t *model = fsdp->model;
    size_t tok_count = (size_t)model->config.vocab_size * model->config.hidden_size;
    size_t out_count = (size_t)model->config.hidden_size * model->config.vocab_size;

    MLLM_RETURN_ON_ERROR(mllm_all_reduce_stream(fsdp->dp_comm,
                                  model->grad_tok_embeddings,
                                  model->grad_tok_embeddings,
                                  tok_count,
                                  MLLM_DTYPE_FLOAT32,
                                  MLLM_REDUCE_OP_SUM,
                                  0, stream));
    MLLM_RETURN_ON_ERROR(mllm_all_reduce_stream(fsdp->dp_comm,
                                  model->grad_output_weight,
                                  model->grad_output_weight,
                                  out_count,
                                  MLLM_DTYPE_FLOAT32,
                                  MLLM_REDUCE_OP_SUM,
                                  0, stream));

    float inv_dp = 1.0f / (float)fsdp->dp_size;
    int block = 256;
    scale_kernel<<<(tok_count + block - 1) / block, block, 0, stream>>>(model->grad_tok_embeddings, tok_count, inv_dp);
    scale_kernel<<<(out_count + block - 1) / block, block, 0, stream>>>(model->grad_output_weight, out_count, inv_dp);
    return MLLM_CUDA_CHECK(cudaGetLastError());
}

int mllm_fsdp_gather_weights(mllm_fsdp_t *fsdp, void *full_weights,
                              cudaStream_t stream) {
    (void)full_weights;
    return MLLM_OK;
}

int mllm_fsdp_scatter_weights(mllm_fsdp_t *fsdp, const void *full_weights,
                               cudaStream_t stream) {
    (void)full_weights;
    return MLLM_OK;
}

int mllm_fsdp_step(mllm_fsdp_t *fsdp,
                   const int *input_ids,
                   const int *position_ids,
                   const float *image_embeds,
                   const int *labels,
                   int batch_size,
                   int seq_len,
                   cudaStream_t stream) {
    mllm_image_batch_t image_batch;
    mllm_image_batch_t *image_batch_ptr = NULL;
    if (image_embeds != NULL && fsdp && fsdp->model) {
        int values_per_image = fsdp_image_values_per_image(&fsdp->model->config);
        if (values_per_image <= 0) return MLLM_ERR_INVALID_INPUT;
        memset(&image_batch, 0, sizeof(image_batch));
        image_batch.data = image_embeds;
        image_batch.batch_size = batch_size;
        image_batch.num_images = fsdp->model->config.num_images;
        image_batch.values_per_image = values_per_image;
        image_batch.location = MLLM_IMAGE_BATCH_HOST_PATCHES_F32;
        image_batch_ptr = &image_batch;
    }

    return mllm_fsdp_step_with_image_batch(fsdp, input_ids, position_ids,
                                           image_batch_ptr, labels,
                                           batch_size, seq_len, stream);
}

int mllm_fsdp_step_with_image_batch(mllm_fsdp_t *fsdp,
                                    const int *input_ids,
                                    const int *position_ids,
                                    const mllm_image_batch_t *image_batch,
                                    const int *labels,
                                    int batch_size,
                                    int seq_len,
                                    cudaStream_t stream) {
    (void)position_ids;
    if (!fsdp || !input_ids || !labels || batch_size <= 0 || seq_len <= 0) {
        return MLLM_ERR_INVALID_INPUT;
    }

    mllm_model_t *model = fsdp->model;
    if (!model) return MLLM_ERR_INVALID_INPUT;

    const float *d_image_embeds = NULL;
    float *owned_d_image_embeds = NULL;
    int *d_input_ids = NULL;
    int *d_labels = NULL;
    float *logits = NULL;
    int rc = MLLM_OK;
    int image_tokens_per_sample = 0;
    int token_count = 0;
    int combined_token_count = 0;
    int vs = 0;
    size_t ids_bytes = 0;
    size_t logits_bytes = 0;
    float loss = 0.0f;
    int expected_values_per_image = fsdp_image_values_per_image(&model->config);
    if (expected_values_per_image <= 0) return MLLM_ERR_INVALID_INPUT;

    if (image_batch && image_batch->data) {
        if (image_batch->batch_size != batch_size ||
            image_batch->num_images != model->config.num_images ||
            image_batch->values_per_image != expected_values_per_image) {
            MLLM_LOG_ERROR("Invalid image batch: got batch=%d num_images=%d values_per_image=%d, expected batch=%d num_images=%d values_per_image=%d",
                           image_batch->batch_size, image_batch->num_images,
                           image_batch->values_per_image, batch_size,
                           model->config.num_images, expected_values_per_image);
            return MLLM_ERR_INVALID_INPUT;
        }
        if (image_batch->location == MLLM_IMAGE_BATCH_HOST_PATCHES_F32) {
            size_t image_count = (size_t)batch_size * image_batch->num_images *
                                 image_batch->values_per_image;
            FSDP_CUDA_GOTO(cudaMalloc(&owned_d_image_embeds,
                                      image_count * sizeof(float)));
            FSDP_CUDA_GOTO(cudaMemcpyAsync(owned_d_image_embeds,
                                           image_batch->data,
                                           image_count * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));
            d_image_embeds = owned_d_image_embeds;
        } else if (image_batch->location == MLLM_IMAGE_BATCH_DEVICE_PATCHES_F32) {
            d_image_embeds = image_batch->data;
        } else {
            MLLM_LOG_ERROR("Invalid image batch location: %d",
                           (int)image_batch->location);
            return MLLM_ERR_INVALID_INPUT;
        }
        image_tokens_per_sample = model->config.num_images;
    }

    token_count = batch_size * seq_len;
    combined_token_count = batch_size * (seq_len + image_tokens_per_sample);
    if (combined_token_count > model->config.max_position_embeddings) {
        MLLM_LOG_ERROR("batch_size * combined_seq_len (%d) exceeds max_position_embeddings (%d)",
                       combined_token_count, model->config.max_position_embeddings);
        cudaFree(owned_d_image_embeds);
        return MLLM_ERR_INVALID_INPUT;
    }
    vs = model->config.vocab_size;

    ids_bytes = (size_t)token_count * sizeof(int);
    logits_bytes = (size_t)combined_token_count * vs * sizeof(float);

    FSDP_CUDA_GOTO(cudaMalloc(&d_input_ids, ids_bytes));
    FSDP_CUDA_GOTO(cudaMalloc(&d_labels, ids_bytes));
    FSDP_CUDA_GOTO(cudaMalloc(&logits, logits_bytes));
    FSDP_CUDA_GOTO(cudaMemcpyAsync(d_input_ids, input_ids, ids_bytes,
                                   cudaMemcpyHostToDevice, stream));
    FSDP_CUDA_GOTO(cudaMemcpyAsync(d_labels, labels, ids_bytes,
                                   cudaMemcpyHostToDevice, stream));

    rc = mllm_model_forward(model, d_input_ids, NULL,
                            d_image_embeds, d_labels, batch_size, seq_len,
                            logits, &loss, stream);
    if (rc != MLLM_OK) goto cleanup;

    rc = mllm_model_backward(model, logits, d_input_ids, d_labels,
                             batch_size, seq_len, image_tokens_per_sample, stream);
    if (rc != MLLM_OK) goto cleanup;

    rc = mllm_fsdp_all_reduce_grads(fsdp, stream);
    if (rc != MLLM_OK) goto cleanup;

    rc = mllm_optimizer_step(fsdp->optimizer, stream);
    if (rc != MLLM_OK) goto cleanup;

    fsdp->step++;
    fsdp->train_loss = loss;

cleanup:
    cudaFree(logits);
    cudaFree(d_labels);
    cudaFree(d_input_ids);
    cudaFree(owned_d_image_embeds);
    return rc;
}
