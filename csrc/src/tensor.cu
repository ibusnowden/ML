// Copyright 2024 mmllm contributors
// Tensor type implementation

#include "tensor.h"
#include "error.h"
#include <cuda_runtime.h>

static size_t dtype_size(mllm_dtype_enum_t dtype) {
    switch (dtype) {
        case MLLM_DTYPE_FP32: return sizeof(float);
        case MLLM_DTYPE_FP16: return sizeof(uint16_t);
        case MLLM_DTYPE_BF16: return sizeof(uint16_t);
    }
    return sizeof(float);
}

int mllm_tensor_alloc(mllm_tensor_t *out, size_t rows, size_t cols,
                      mllm_dtype_enum_t dtype) {
    size_t bytes = rows * cols * dtype_size(dtype);
    MLLM_CUDA_CHECK(cudaMalloc(&out->ptr, bytes));
    out->rows = rows;
    out->cols = cols;
    out->dtype = dtype;
    return MLLM_OK;
}

void mllm_tensor_free(mllm_tensor_t *t) {
    if (t && t->ptr) {
        cudaFree((void*)t->ptr);
        t->ptr = 0;
    }
}

int mllm_tensor_copy_from_host(mllm_tensor_t *dst, const void *src) {
    size_t bytes = dst->rows * dst->cols * dtype_size(dst->dtype);
    return MLLM_CUDA_CHECK(cudaMemcpy(dst->ptr, src, bytes, cudaMemcpyHostToDevice));
}

int mllm_tensor_copy_to_host(const mllm_tensor_t *src, void *dst) {
    size_t bytes = src->rows * src->cols * dtype_size(src->dtype);
    return MLLM_CUDA_CHECK(cudaMemcpy(dst, (const void*)src->ptr, bytes, cudaMemcpyDeviceToHost));
}

int mllm_tensor_zero(mllm_tensor_t *t) {
    size_t bytes = t->rows * t->cols * dtype_size(t->dtype);
    return MLLM_CUDA_CHECK(cudaMemset((void*)t->ptr, 0, bytes));
}
