// Copyright 2024 entropy contributors
// Checkpoint save/restore for FSDP training

#pragma once

#include "fsdp.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Checkpoint metadata
typedef struct {
    int64_t step;
    float loss;
    int64_t tokens_seen;
    int version;  // checkpoint format version
} mllm_checkpoint_meta_t;

// Save checkpoint (sharded across ranks)
// Writes one file per rank + a shared metadata file
int mllm_checkpoint_save(mllm_fsdp_t *fsdp, const char *dir,
                         int64_t step, cudaStream_t stream);

// Load checkpoint
// Returns MLLM_OK on success, MLLM_ERR_NOT_READY if no checkpoint found
int mllm_checkpoint_load(mllm_fsdp_t *fsdp, const char *dir,
                         int64_t step, cudaStream_t stream,
                         mllm_checkpoint_meta_t *meta);

#ifdef __cplusplus
}
#endif
