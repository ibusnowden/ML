// Copyright 2024 mmllm contributors
// CUDA stream management for compute-communication overlap

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Create a named CUDA stream (non-blocking)
int mllm_create_stream(const char *name, cudaStream_t *out_stream);

// Destroy a named CUDA stream
int mllm_destroy_stream(cudaStream_t stream);

// Synchronize a stream
int mllm_stream_sync(cudaStream_t stream);

// Stream-based event record/retrieve for overlap tracking
int mllm_stream_record_event(cudaStream_t stream, cudaEvent_t *out_event);
int mllm_stream_wait_event(cudaStream_t stream, cudaEvent_t event, unsigned int flags);

// Synchronize an event
int mllm_event_sync(cudaEvent_t event);

// Destroy an event
int mllm_event_destroy(cudaEvent_t event);

#ifdef __cplusplus
}
#endif
