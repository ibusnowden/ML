// Copyright 2024 entropy contributors
// Model parallel topology implementation

#include "comm.h"
#include "error.h"
#include "nccl_wrapper.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

static int mllm_topology_set_ranks(mllm_mp_topology_t *topo, 
                                    const char *host_id,
                                    int node_id, int rank, int world_size,
                                    int tp_size, int pp_size) {
    topo->tp_size = tp_size;
    topo->pp_size = pp_size;
    topo->dp_size = world_size / (tp_size * pp_size);
    topo->rank = rank;
    topo->world_size = world_size;
    topo->tp_mode = MLLM_TP_NONE;
    
    // Initialize compute stream
    MLLM_CUDA_CHECK(cudaStreamCreate(&topo->compute_stream));
    
    // Create communication streams
    topo->num_comm_streams = tp_size;
    topo->comm_streams = (cudaStream_t*)malloc(tp_size * sizeof(cudaStream_t));
    for (int i = 0; i < tp_size; i++) {
        MLLM_CUDA_CHECK(cudaStreamCreateWithFlags(&topo->comm_streams[i], 
                                                   cudaStreamNonBlocking));
    }
    return MLLM_OK;
}

int mllm_topology_init(mllm_mp_topology_t *topo,
                       const char *host_id,
                       int node_id,
                       int rank,
                       int world_size,
                       int tp_size,
                       int pp_size) {
    MLLM_CHECK(topo != NULL, "topo is NULL");
    MLLM_CHECK(host_id != NULL, "host_id is NULL");
    
    MLLM_RETURN_ON_ERROR(mllm_topology_set_ranks(topo, host_id, node_id, rank, world_size, 
                           tp_size, pp_size));
    
    // Initialize data parallel group (all GPUs)
    topo->data_parallel_comm.rank = rank;
    topo->data_parallel_comm.world_size = world_size;
    topo->data_parallel_comm.local_rank = rank % 8; // assuming 8 GPUs per node
    topo->data_parallel_comm.local_size = 8;
    MLLM_CHECK(mllm_nccl_init(&topo->data_parallel_comm, 
                              host_id, node_id) == MLLM_OK, 
              "Failed to initialize data parallel group");
    MLLM_CUDA_CHECK(cudaStreamCreate(&topo->data_parallel_comm.stream));
    
    // Initialize tensor parallel group (per TP group)
    int tp_rank = rank % tp_size;
    int tp_group_id = rank / tp_size;
    topo->tensor_parallel_comm.rank = tp_rank;
    topo->tensor_parallel_comm.world_size = tp_size;
    topo->tensor_parallel_comm.local_rank = tp_rank;
    topo->tensor_parallel_comm.local_size = tp_size;
    MLLM_CHECK(mllm_nccl_init(&topo->tensor_parallel_comm,
                              host_id, node_id) == MLLM_OK,
              "Failed to initialize tensor parallel group");
    MLLM_CUDA_CHECK(cudaStreamCreate(&topo->tensor_parallel_comm.stream));
    
    // Initialize pipeline parallel group (per PP stage)
    int pp_rank = (rank / tp_size) % pp_size;
    int pp_group_id = rank / (tp_size * pp_size);
    topo->pipeline_parallel_comm.rank = pp_rank;
    topo->pipeline_parallel_comm.world_size = pp_size;
    topo->pipeline_parallel_comm.local_rank = pp_rank;
    topo->pipeline_parallel_comm.local_size = pp_size;
    MLLM_CHECK(mllm_nccl_init(&topo->pipeline_parallel_comm,
                              host_id, node_id) == MLLM_OK,
              "Failed to initialize pipeline parallel group");
    MLLM_CUDA_CHECK(cudaStreamCreate(&topo->pipeline_parallel_comm.stream));
    
    return MLLM_OK;
}

void mllm_topology_destroy(mllm_mp_topology_t *topo) {
    if (topo == NULL) return;
    
    mllm_nccl_destroy(&topo->data_parallel_comm);
    mllm_nccl_destroy(&topo->tensor_parallel_comm);
    mllm_nccl_destroy(&topo->pipeline_parallel_comm);
    
    if (topo->compute_stream != NULL) {
        cudaStreamDestroy(topo->compute_stream);
    }
    
    if (topo->comm_streams != NULL) {
        for (int i = 0; i < topo->num_comm_streams; i++) {
            if (topo->comm_streams[i] != NULL) {
                cudaStreamDestroy(topo->comm_streams[i]);
            }
        }
        free(topo->comm_streams);
        topo->comm_streams = NULL;
    }
}

int mllm_grad_all_reduce(mllm_mp_topology_t *topo,
                         void *grad_buf,
                         size_t element_count,
                         mllm_dtype_t dtype) {
    MLLM_CHECK(topo != NULL, "topo is NULL");
    MLLM_CHECK(grad_buf != NULL, "grad_buf is NULL");
    
    // In-place all-reduce on gradients with compute stream overlap
    return mllm_all_reduce_stream(&topo->data_parallel_comm,
                                 grad_buf, grad_buf,
                                 element_count, dtype,
                                 MLLM_REDUCE_OP_SUM,
                                 0, topo->compute_stream);
}

int mllm_tp_sync_param(mllm_mp_topology_t *topo,
                       void *param_buf,
                       size_t element_count,
                       mllm_dtype_t dtype) {
    MLLM_CHECK(topo != NULL, "topo is NULL");
    MLLM_CHECK(param_buf != NULL, "param_buf is NULL");
    
    // Sync parameters across tensor parallel group (average them)
    return mllm_all_reduce_stream(&topo->tensor_parallel_comm,
                                 param_buf, param_buf,
                                 element_count, dtype,
                                 MLLM_REDUCE_OP_SUM,
                                 0, topo->compute_stream);
}

int mllm_pp_send(mllm_mp_topology_t *topo,
                 const void *buf,
                 size_t num_bytes) {
    MLLM_CHECK(topo != NULL, "topo is NULL");
    MLLM_CHECK(buf != NULL, "buf is NULL");
    
    // Send activation to next stage
    // Calculate the rank of the next stage
    int current_pp_rank = topo->pipeline_parallel_comm.rank;
    int next_pp_rank = (current_pp_rank + 1) % topo->pp_size;
    
    return mllm_send(&topo->pipeline_parallel_comm, buf, num_bytes,
                    MLLM_DTYPE_FLOAT32, next_pp_rank);
}

int mllm_pp_recv(mllm_mp_topology_t *topo,
                 void *buf,
                 size_t num_bytes) {
    MLLM_CHECK(topo != NULL, "topo is NULL");
    MLLM_CHECK(buf != NULL, "buf is NULL");
    
    // Receive activation from previous stage
    int current_pp_rank = topo->pipeline_parallel_comm.rank;
    int prev_pp_rank = (current_pp_rank - 1 + topo->pp_size) % topo->pp_size;
    
    return mllm_recv(&topo->pipeline_parallel_comm, buf, num_bytes,
                    MLLM_DTYPE_FLOAT32, prev_pp_rank);
}

int mllm_barrier(mllm_mp_topology_t *topo) {
    MLLM_CHECK(topo != NULL, "topo is NULL");
    
    // Synchronize all ranks
    MLLM_CUDA_CHECK(cudaDeviceSynchronize());
    return MLLM_OK;
}

int mllm_synchronize_all(mllm_mp_topology_t *topo) {
    MLLM_CHECK(topo != NULL, "topo is NULL");
    
    // Synchronize all CUDA streams
    MLLM_CUDA_CHECK(cudaDeviceSynchronize());
    
    // Also synchronize all communication streams
    for (int i = 0; i < topo->num_comm_streams; i++) {
        MLLM_CUDA_CHECK(cudaStreamSynchronize(topo->comm_streams[i]));
    }
    
    return MLLM_OK;
}
