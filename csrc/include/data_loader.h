// Copyright 2024 entropy contributors
// Data loader for training (text + multimodal)

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Training batch
typedef struct {
    int *input_ids;       // [batch_size, seq_len]
    int *position_ids;    // [batch_size, seq_len] — generated, not from data file
    int *labels;          // [batch_size, seq_len] — separate from input_ids
    float *image_embeds;  // host patch-major float32 image data (optional)
    int batch_size;
    int seq_len;
    int num_images;
    int vision_hidden_size;
    bool has_vision;      // whether this batch has image data
} mllm_batch_t;

// Data loader state
typedef struct {
    int *data;           // Flattened token data
    size_t data_size;    // Total tokens in dataset
    int batch_size;
    int seq_len;
    int num_images;
    int vision_hidden_size;
    int64_t offset;      // Current position in dataset
    int64_t total_batches;
    bool shuffle;
    uint64_t seed;
} mllm_data_loader_t;

// Create data loader from tokenized file
// data_path: path to a file with uint32 tokens (one per line or binary)
int mllm_data_loader_create(mllm_data_loader_t *loader,
                            const char *data_path,
                            int batch_size, int seq_len,
                            int num_images, int vision_hidden_size,
                            bool shuffle, uint64_t seed);

// Destroy data loader
void mllm_data_loader_destroy(mllm_data_loader_t *loader);

// Get next batch (returns MLLM_OK on success, MLLM_ERR_NOT_READY at end)
int mllm_data_loader_next(mllm_data_loader_t *loader, mllm_batch_t *batch);

// Reset loader to beginning
void mllm_data_loader_reset(mllm_data_loader_t *loader);

// Check if loader is exhausted
bool mllm_data_loader_is_exhausted(mllm_data_loader_t *loader);

#ifdef __cplusplus
}
#endif
