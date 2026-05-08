// Copyright 2024 mmllm contributors
// Tensor type definitions for mmllm

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Supported dtypes
typedef enum {
    MLLM_DTYPE_FP32,
    MLLM_DTYPE_FP16,
    MLLM_DTYPE_BF16,
} mllm_dtype_enum_t;

// Compute capability helpers
typedef struct {
    uint64_t ptr;       // Device pointer
    mllm_dtype_enum_t dtype;
    size_t rows;
    size_t cols;
} mllm_tensor_t;

// Allocate tensor on GPU
int mllm_tensor_alloc(mllm_tensor_t *out, size_t rows, size_t cols,
                      mllm_dtype_enum_t dtype);

// Free tensor memory
void mllm_tensor_free(mllm_tensor_t *t);

// Copy host to device
int mllm_tensor_copy_from_host(mllm_tensor_t *dst, const void *src);

// Copy device to host
int mllm_tensor_copy_to_host(const mllm_tensor_t *src, void *dst);

// Zero tensor
int mllm_tensor_zero(mllm_tensor_t *t);

#ifdef __cplusplus
}
#endif
