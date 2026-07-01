// Copyright 2024 entropy contributors
// Error handling and return code utilities

#pragma once

#include <stdint.h>
#include <string.h>
#include <cstdarg>
#include "logging.h"

#ifdef __cplusplus
extern "C" {
#endif

// Common return codes
typedef enum {
    MLLM_OK = 0,
    MLLM_ERR_UNINITIALIZED = -1,
    MLLM_ERR_NCCL = -2,
    MLLM_ERR_CUDA = -3,
    MLLM_ERR_INVALID_INPUT = -4,
    MLLM_ERR_INVALID_RANK = -5,
    MLLM_ERR_ALLOC = -6,
    MLLM_ERR_TIMEOUT = -7,
    MLLM_ERR_COMM_FAIL = -8,
    MLLM_ERR_STREAM = -9,
    MLLM_ERR_NOT_READY = -10,
} mllm_error_t;

// Convert mllm_error_t to string
const char* mllm_error_str(int err);

// Log an error and return it
static inline mllm_error_t mllm_error_with_msg(mllm_error_t err, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    MLLM_LOG_ERROR("%s: ", mllm_error_str(err));
    vfprintf(stderr, fmt, args);
    va_end(args);
    return err;
}

// Check an error and return early if non-zero
#define MLLM_RETURN_ON_ERROR(err) \
    do { \
        if ((err) != MLLM_OK) { \
            return (err); \
        } \
    } while (0)

// Check CUDA error
#define MLLM_CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            MLLM_LOG_ERROR("CUDA error %s at %s:%d", \
                          cudaGetErrorString(err), __FILE__, __LINE__); \
            return MLLM_ERR_CUDA; \
        } \
    } while (0)

// Check NCCL error
#define MLLM_NCCL_CHECK(call) \
    do { \
        ncclResult_t err = (call); \
        if (err != ncclSuccess) { \
            MLLM_LOG_ERROR("NCCL error %s at %s:%d", \
                          ncclGetErrorString(err), __FILE__, __LINE__); \
            return MLLM_ERR_NCCL; \
        } \
    } while (0)

#ifdef __cplusplus
}
#endif
