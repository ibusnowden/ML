// Copyright 2024 mmllm contributors
// Error handling implementation

#include "error.h"
#include "logging.h"
#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdarg>

static const char* mllm_error_strings[] = {
    "OK",
    "Uninitialized",
    "NCCL error",
    "CUDA error",
    "Invalid input",
    "Invalid rank",
    "Allocation failed",
    "Timeout",
    "Communication failure",
    "Stream error",
    "Not ready",
};

const char* mllm_error_str(int err) {
    int idx = (int)(-(int)err);
    if (idx >= 0 && idx < (int)(sizeof(mllm_error_strings) / sizeof(mllm_error_strings[0]))) {
        return mllm_error_strings[idx];
    }
    return "Unknown error";
}
