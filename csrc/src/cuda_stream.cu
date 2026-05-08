// Copyright 2024 mmllm contributors
// CUDA stream management implementation

#include "cuda_stream.h"
#include "error.h"
#include <cuda_runtime.h>

int mllm_create_stream(const char *name, cudaStream_t *out_stream) {
    (void)name; // stream name is informational, not exposed by CUDA
    MLLM_CUDA_CHECK(cudaStreamCreateWithFlags(out_stream, cudaStreamNonBlocking));
    return MLLM_OK;
}

int mllm_destroy_stream(cudaStream_t stream) {
    return MLLM_CUDA_CHECK(cudaStreamDestroy(stream));
}

int mllm_stream_sync(cudaStream_t stream) {
    return MLLM_CUDA_CHECK(cudaStreamSynchronize(stream));
}

int mllm_stream_record_event(cudaStream_t stream, cudaEvent_t *out_event) {
    MLLM_CUDA_CHECK(cudaEventCreate(out_event));
    return MLLM_CUDA_CHECK(cudaEventRecord(*out_event, stream));
}

int mllm_stream_wait_event(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    return MLLM_CUDA_CHECK(cudaStreamWaitEvent(stream, event, flags));
}

int mllm_event_sync(cudaEvent_t event) {
    return MLLM_CUDA_CHECK(cudaEventSynchronize(event));
}

int mllm_event_destroy(cudaEvent_t event) {
    return MLLM_CUDA_CHECK(cudaEventDestroy(event));
}
