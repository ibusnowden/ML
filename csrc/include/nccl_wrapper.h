// Copyright 2024 entropy contributors
// NCCL wrapper for distributed communication primitives

#pragma once

#include <nccl.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum number of GPUs supported per node
#define MLLM_MAX_GPUS 8
// Maximum number of nodes
#define MLLM_MAX_NODES 16
// Total max GPUs across all nodes
#define MLLM_MAX_TOTAL_GPUS (MLLM_MAX_NODES * MLLM_MAX_GPUS)

// NCCL communication primitive types
typedef enum {
    MLLM_REDUCE_OP_SUM = 0,
    MLLM_REDUCE_OP_PROD,
    MLLM_REDUCE_OP_MIN,
    MLLM_REDUCE_OP_MAX,
} mllm_reduce_op_t;

// NCCL data types
typedef enum {
    MLLM_DTYPE_FLOAT32 = 0,
    MLLM_DTYPE_FLOAT16,
    MLLM_DTYPE_BFLOAT16,
    MLLM_DTYPE_INT32,
    MLLM_DTYPE_INT64,
} mllm_dtype_t;

// Communication group
typedef struct {
    ncclComm_t nccl_comm;
    int rank;
    int world_size;
    int local_rank;
    int local_size;
    int node_id;
    cudaStream_t stream;  // Master stream for NCCL operations
    char host_id[256];    // Node identifier for NCCL
} mllm_comm_group_t;

// NCCL communicator initialization
// Returns 0 on success
int mllm_nccl_init(mllm_comm_group_t *comm, const char *host_id, int node_id);

// Destroy communicator
void mllm_nccl_destroy(mllm_comm_group_t *comm);

// Get NCCL datatype from mllm dtype
ncclDataType_t mllm_dtype_to_nccl(mllm_dtype_t dtype);

// Get NCCL reduce op from mllm reduce op
ncclRedOp_t mllm_reduce_op_to_nccl(mllm_reduce_op_t op);

// ============== Collective Operations ==============

// All-reduce with in-place or out-of-place gradient synchronization
// result_buf is overwritten with reduced values; result_buf == input_buf for in-place
int mllm_all_reduce(mllm_comm_group_t *comm,
                    void *input_buf,
                    void *output_buf,
                    size_t element_count,
                    mllm_dtype_t dtype,
                    mllm_reduce_op_t op);

// All-reduce with CUDA stream (non-blocking)
int mllm_all_reduce_stream(mllm_comm_group_t *comm,
                           void *input_buf,
                           void *output_buf,
                           size_t element_count,
                           mllm_dtype_t dtype,
                           mllm_reduce_op_t op,
                           int root,
                           cudaStream_t stream);

// All-gather: gather tensors from all ranks to all ranks
// Each rank sends send_buf (size_t send_count elements) and receives recv_buf
int mllm_all_gather(mllm_comm_group_t *comm,
                    const void *send_buf,
                    void *recv_buf,
                    size_t send_count,
                    mllm_dtype_t dtype);

// All-gather with CUDA stream
int mllm_all_gather_stream(mllm_comm_group_t *comm,
                           const void *send_buf,
                           void *recv_buf,
                           size_t send_count,
                           mllm_dtype_t dtype,
                           cudaStream_t stream);

// Broadcast: root tensor to all other ranks
int mllm_broadcast(mllm_comm_group_t *comm,
                   void *buffer,
                   size_t element_count,
                   mllm_dtype_t dtype,
                   int root);

// Broadcast with CUDA stream
int mllm_broadcast_stream(mllm_comm_group_t *comm,
                          void *buffer,
                          size_t element_count,
                          mllm_dtype_t dtype,
                          int root,
                          cudaStream_t stream);

// Reduce: results from all ranks to root
int mllm_reduce(mllm_comm_group_t *comm,
                void *input_buf,
                void *output_buf,
                size_t element_count,
                mllm_dtype_t dtype,
                mllm_reduce_op_t op,
                int root);

// Reduce with CUDA stream
int mllm_reduce_stream(mllm_comm_group_t *comm,
                       void *input_buf,
                       void *output_buf,
                       size_t element_count,
                       mllm_dtype_t dtype,
                       mllm_reduce_op_t op,
                       int root,
                       cudaStream_t stream);

// Send/Receive point-to-point operations
int mllm_send(mllm_comm_group_t *comm,
              const void *buf,
              size_t element_count,
              mllm_dtype_t dtype,
              int dest);

int mllm_recv(mllm_comm_group_t *comm,
              void *buf,
              size_t element_count,
              mllm_dtype_t dtype,
              int src);

// ============== Stream & Sync ==============

// Synchronize all ranks on a stream
int mllm_stream_synchronize(mllm_comm_group_t *comm, cudaStream_t stream);

// Create a non-blocking CUDA stream for communication overlap
int mllm_create_comm_stream(mllm_comm_group_t *comm, cudaStream_t *out_stream);

// Destroy a custom stream
void mllm_destroy_stream(cudaStream_t stream);

// ============== Topology & Info ==============

// Get NCCL version string
const char* mllm_nccl_version(void);

// Check if NCCL is available
bool mllm_nccl_available(void);

#ifdef __cplusplus
}
#endif
