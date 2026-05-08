// Copyright 2024 mmllm contributors
// Communication group abstraction for multi-GPU / multi-node coordination

#pragma once

#include "nccl_wrapper.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Tensor parallelism mode
typedef enum {
    MLLM_TP_NONE = 0,       // No tensor parallelism
    MLLM_TP_COLUMN,          // Column-wise (attention QKV split)
    MLLM_TP_ROW,             // Row-wise (attention output, MLP intermediate)
    MLLM_TP_EXPERT,          // Expert parallelism for MoE
} mllm_tp_mode_t;

// Pipeline parallelism stage info
typedef struct {
    int stage_id;
    int num_stages;
    int rank;
    int prev_stage;  // rank of previous pipeline stage (-1 if first)
    int next_stage;  // rank of next pipeline stage (-1 if last)
} mllm_pp_stage_t;

// Full model parallel topology
typedef struct {
    mllm_comm_group_t data_parallel_comm;   // Data parallel group (all GPUs)
    mllm_comm_group_t tensor_parallel_comm; // Tensor parallel group (per TP group)
    mllm_comm_group_t pipeline_parallel_comm; // Pipeline parallel group (per PP stage)
    
    int tp_size;      // Tensor parallel degree
    int pp_size;      // Pipeline parallel degree
    int dp_size;      // Data parallel degree
    int rank;         // Global rank
    int world_size;   // Total GPUs
    
    mllm_tp_mode_t tp_mode;
    
    // Per-GPU CUDA streams for communication overlap
    cudaStream_t compute_stream;
    cudaStream_t *comm_streams;  // Array of tp_size streams
    int num_comm_streams;
} mllm_mp_topology_t;

// Initialize the full model parallel topology
int mllm_topology_init(mllm_mp_topology_t *topo,
                       const char *host_id,
                       int node_id,
                       int rank,
                       int world_size,
                       int tp_size,
                       int pp_size);

// Clean up topology
void mllm_topology_destroy(mllm_mp_topology_t *topo);

// Execute an all-reduce on gradients with proper stream ordering
int mllm_grad_all_reduce(mllm_mp_topology_t *topo,
                         void *grad_buf,
                         size_t element_count,
                         mllm_dtype_t dtype);

// Synchronize a model parameter across tensor parallel group
int mllm_tp_sync_param(mllm_mp_topology_t *topo,
                       void *param_buf,
                       size_t element_count,
                       mllm_dtype_t dtype);

// Pipeline: send activation to next stage
int mllm_pp_send(mllm_mp_topology_t *topo,
                 const void *buf,
                 size_t num_bytes);

// Pipeline: recv activation from previous stage
int mllm_pp_recv(mllm_mp_topology_t *topo,
                 void *buf,
                 size_t num_bytes);

// Barrier across all ranks
int mllm_barrier(mllm_mp_topology_t *topo);

// Synchronize all streams
int mllm_synchronize_all(mllm_mp_topology_t *topo);

#ifdef __cplusplus
}
#endif
