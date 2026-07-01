// Copyright 2024 entropy contributors
// Data loader for training (text + multimodal)

#include "data_loader.h"
#include "error.h"
#include "logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Simple LCG random number generator
typedef struct {
    uint64_t state;
} lcg_rng_t;

static void lcg_init(lcg_rng_t *rng, uint64_t seed) {
    rng->state = seed;
}

static uint64_t lcg_next(lcg_rng_t *rng) {
    rng->state = rng->state * 6364136223846793005ULL + 1442695040888963407ULL;
    return rng->state;
}

static int lcg_rand(lcg_rng_t *rng, int max) {
    return (int)(lcg_next(rng) % max);
}

// Fisher-Yates shuffle on token array
static void shuffle_tokens(int *data, size_t size, lcg_rng_t *rng) {
    for (size_t i = size - 1; i > 0; i--) {
        int j = lcg_rand(rng, (int)(i + 1));
        int temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }
}

int mllm_data_loader_create(mllm_data_loader_t *loader,
                            const char *data_path,
                            int batch_size, int seq_len,
                            int num_images, int vision_hidden_size,
                            bool shuffle, uint64_t seed) {
    memset(loader, 0, sizeof(*loader));
    loader->batch_size = batch_size;
    loader->seq_len = seq_len;
    loader->num_images = num_images;
    loader->vision_hidden_size = vision_hidden_size;
    loader->shuffle = shuffle;
    loader->seed = seed;

    // Load tokens from file. Use .bin/.tokens binary files as uint32_t;
    // otherwise parse one unsigned token per text field.
    const char *ext = strrchr(data_path, '.');
    bool binary = ext && (strcmp(ext, ".bin") == 0 || strcmp(ext, ".tokens") == 0);
    FILE *f = fopen(data_path, binary ? "rb" : "r");
    if (!f) {
        fprintf(stderr, "Failed to open data file: %s\n", data_path);
        return MLLM_ERR_ALLOC;
    }

    if (binary) {
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, 0, SEEK_SET);

        loader->data_size = size / sizeof(uint32_t);
        loader->data = (int*)malloc(loader->data_size * sizeof(int));
        if (!loader->data) {
            fclose(f);
            return MLLM_ERR_ALLOC;
        }
        uint32_t *tmp = (uint32_t*)malloc(loader->data_size * sizeof(uint32_t));
        if (!tmp) {
            fclose(f);
            free(loader->data);
            return MLLM_ERR_ALLOC;
        }
        fread(tmp, sizeof(uint32_t), loader->data_size, f);
        for (size_t i = 0; i < loader->data_size; i++) loader->data[i] = (int)tmp[i];
        free(tmp);
    } else {
        int count = 0;
        uint32_t tok;
        while (fscanf(f, "%u", &tok) == 1) count++;
        rewind(f);

        loader->data = (int*)malloc((size_t)count * sizeof(int));
        if (!loader->data) {
            fclose(f);
            return MLLM_ERR_ALLOC;
        }
        loader->data_size = count;
        for (int i = 0; i < count; i++) {
            fscanf(f, "%u", &loader->data[i]);
        }
    }
    fclose(f);

    // Shuffle if requested
    if (loader->shuffle && loader->data_size > 0) {
        lcg_rng_t rng;
        lcg_init(&rng, seed);
        shuffle_tokens(loader->data, loader->data_size, &rng);
        MLLM_LOG_INFO("Data shuffled with seed %lu", (unsigned long)seed);
    }

    loader->total_batches = (loader->data_size - 1) / (batch_size * seq_len);
    MLLM_LOG_INFO("Data loader created: %zu tokens, batch_size=%d, seq_len=%d, %ld batches",
                  loader->data_size, batch_size, seq_len, (long)loader->total_batches);
    return MLLM_OK;
}

void mllm_data_loader_destroy(mllm_data_loader_t *loader) {
    if (loader && loader->data) {
        free(loader->data);
        loader->data = NULL;
    }
    memset(loader, 0, sizeof(*loader));
}

int mllm_data_loader_next(mllm_data_loader_t *loader, mllm_batch_t *batch) {
    if (loader->offset >= loader->total_batches) {
        return MLLM_ERR_NOT_READY;
    }

    int64_t start = loader->offset * loader->batch_size * loader->seq_len;
    if (start + (size_t)(loader->batch_size * loader->seq_len) > loader->data_size) {
        return MLLM_ERR_NOT_READY;
    }

    batch->batch_size = loader->batch_size;
    batch->seq_len = loader->seq_len;
    batch->num_images = loader->num_images;
    batch->vision_hidden_size = loader->vision_hidden_size;
    batch->input_ids = loader->data + start;
    batch->has_vision = false;
    batch->image_embeds = NULL;

    // Generate proper position IDs: [0, 1, 2, ..., seq_len-1]
    // Allocate position IDs per batch (caller must free)
    batch->position_ids = (int*)malloc(loader->batch_size * loader->seq_len * sizeof(int));
    if (!batch->position_ids) {
        return MLLM_ERR_ALLOC;
    }
    for (int b = 0; b < loader->batch_size; b++) {
        int *pos = batch->position_ids + b * loader->seq_len;
        for (int i = 0; i < loader->seq_len; i++) {
            pos[i] = i;
        }
    }

    // Labels are shifted for next-token prediction within each sample.
    batch->labels = (int*)malloc(loader->batch_size * loader->seq_len * sizeof(int));
    if (!batch->labels) {
        free(batch->position_ids);
        batch->position_ids = NULL;
        return MLLM_ERR_ALLOC;
    }
    for (int b = 0; b < loader->batch_size; b++) {
        int *lbl = batch->labels + b * loader->seq_len;
        int *inp = batch->input_ids + b * loader->seq_len;
        for (int i = 0; i < loader->seq_len - 1; i++) {
            lbl[i] = inp[i + 1];
        }
        lbl[loader->seq_len - 1] = inp[loader->seq_len - 1];
    }

    loader->offset++;
    return MLLM_OK;
}

void mllm_data_loader_reset(mllm_data_loader_t *loader) {
    loader->offset = 0;
}

bool mllm_data_loader_is_exhausted(mllm_data_loader_t *loader) {
    return loader->offset >= loader->total_batches;
}
