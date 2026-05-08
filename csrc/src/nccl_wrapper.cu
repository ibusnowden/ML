// Copyright 2024 mmllm contributors
// NCCL wrapper implementation

#include "nccl_wrapper.h"
#include "error.h"
#include "logging.h"
#include <string.h>

int mllm_nccl_init(mllm_comm_group_t *comm, const char *host_id, int node_id) {
    MLLM_CHECK(comm != NULL, "comm is NULL");
    MLLM_CHECK(host_id != NULL, "host_id is NULL");

    (void)node_id;
    ncclComm_t nccl_comm = nullptr;
    ncclUniqueId unique_id;
    ncclResult_t rc = ncclGetUniqueId(&unique_id);
    if (rc != ncclSuccess) {
        MLLM_LOG_ERROR("ncclGetUniqueId failed: %s", ncclGetErrorString(rc));
        return MLLM_ERR_NCCL;
    }
    rc = ncclCommInitRank(&nccl_comm, comm->world_size, unique_id, comm->rank);
    if (rc != ncclSuccess) {
        MLLM_LOG_ERROR("ncclCommInitRank failed: %s", ncclGetErrorString(rc));
        return MLLM_ERR_NCCL;
    }

    comm->nccl_comm = nccl_comm;
    comm->node_id = node_id;
    strncpy(comm->host_id, host_id, sizeof(comm->host_id) - 1);
    comm->host_id[sizeof(comm->host_id) - 1] = '\0';

    MLLM_LOG_INFO("NCCL rank %d world_size %d initialized", node_id, comm->world_size);
    return MLLM_OK;
}

void mllm_nccl_destroy(mllm_comm_group_t *comm) {
    if (comm == NULL) return;
    if (comm->nccl_comm) {
        ncclCommDestroy(comm->nccl_comm);
        comm->nccl_comm = nullptr;
    }
}

ncclDataType_t mllm_dtype_to_nccl(mllm_dtype_t dtype) {
    switch (dtype) {
        case MLLM_DTYPE_FLOAT32:  return ncclFloat32;
        case MLLM_DTYPE_FLOAT16:  return ncclFloat16;
        case MLLM_DTYPE_BFLOAT16: return ncclBfloat16;
        case MLLM_DTYPE_INT32:    return ncclInt32;
        case MLLM_DTYPE_INT64:    return ncclInt64;
        default:                   return ncclFloat32;
    }
}

ncclRedOp_t mllm_reduce_op_to_nccl(mllm_reduce_op_t op) {
    switch (op) {
        case MLLM_REDUCE_OP_SUM:   return ncclSum;
        case MLLM_REDUCE_OP_PROD:  return ncclProd;
        case MLLM_REDUCE_OP_MIN:   return ncclMin;
        case MLLM_REDUCE_OP_MAX:   return ncclMax;
        default:                   return ncclSum;
    }
}

int mllm_all_reduce(mllm_comm_group_t *comm,
                    void *input_buf,
                    void *output_buf,
                    size_t element_count,
                    mllm_dtype_t dtype,
                    mllm_reduce_op_t op) {
    MLLM_CHECK(comm != NULL, "comm is NULL");
    MLLM_CHECK(input_buf != NULL, "input_buf is NULL");
    MLLM_CHECK(output_buf != NULL, "output_buf is NULL");

    ncclDataType_t nccl_dtype = mllm_dtype_to_nccl(dtype);
    ncclRedOp_t nccl_op = mllm_reduce_op_to_nccl(op);

    ncclResult_t rc = ncclAllReduce(input_buf, output_buf, element_count,
                                     nccl_dtype, nccl_op, comm->nccl_comm, 0);
    if (rc != ncclSuccess) {
        MLLM_LOG_ERROR("ncclAllReduce failed: %s", ncclGetErrorString(rc));
        return MLLM_ERR_NCCL;
    }
    return MLLM_OK;
}

int mllm_all_reduce_stream(mllm_comm_group_t *comm,
                           void *input_buf,
                           void *output_buf,
                           size_t element_count,
                           mllm_dtype_t dtype,
                           mllm_reduce_op_t op,
                           int root,
                           cudaStream_t stream) {
    (void)root;
    MLLM_CHECK(comm != NULL, "comm is NULL");
    MLLM_CHECK(input_buf != NULL, "input_buf is NULL");
    MLLM_CHECK(output_buf != NULL, "output_buf is NULL");

    ncclDataType_t nccl_dtype = mllm_dtype_to_nccl(dtype);
    ncclRedOp_t nccl_op = mllm_reduce_op_to_nccl(op);

    ncclResult_t rc = ncclAllReduce(input_buf, output_buf, element_count,
                                     nccl_dtype, nccl_op, comm->nccl_comm, stream);
    if (rc != ncclSuccess) {
        MLLM_LOG_ERROR("ncclAllReduce (stream) failed: %s", ncclGetErrorString(rc));
        return MLLM_ERR_NCCL;
    }
    return MLLM_OK;
}

int mllm_all_gather(mllm_comm_group_t *comm,
                    const void *send_buf,
                    void *recv_buf,
                    size_t send_count,
                    mllm_dtype_t dtype) {
    MLLM_CHECK(comm != NULL, "comm is NULL");
    MLLM_CHECK(send_buf != NULL, "send_buf is NULL");
    MLLM_CHECK(recv_buf != NULL, "recv_buf is NULL");

    ncclDataType_t nccl_dtype = mllm_dtype_to_nccl(dtype);
    ncclResult_t rc = ncclAllGather(send_buf, recv_buf, send_count,
                                     nccl_dtype, comm->nccl_comm, 0);
    if (rc != ncclSuccess) {
        MLLM_LOG_ERROR("ncclAllGather failed: %s", ncclGetErrorString(rc));
        return MLLM_ERR_NCCL;
    }
    return MLLM_OK;
}

int mllm_all_gather_stream(mllm_comm_group_t *comm,
                            const void *send_buf,
                            void *recv_buf,
                            size_t send_count,
                            mllm_dtype_t dtype,
                            cudaStream_t stream) {
    MLLM_CHECK(comm != NULL, "comm is NULL");
    MLLM_CHECK(send_buf != NULL, "send_buf is NULL");
    MLLM_CHECK(recv_buf != NULL, "recv_buf is NULL");

    ncclDataType_t nccl_dtype = mllm_dtype_to_nccl(dtype);
    ncclResult_t rc = ncclAllGather(send_buf, recv_buf, send_count,
                                     nccl_dtype, comm->nccl_comm, stream);
    if (rc != ncclSuccess) {
        MLLM_LOG_ERROR("ncclAllGather (stream) failed: %s", ncclGetErrorString(rc));
        return MLLM_ERR_NCCL;
    }
    return MLLM_OK;
}

int mllm_broadcast(mllm_comm_group_t *comm,
                   void *buffer,
                   size_t element_count,
                   mllm_dtype_t dtype,
                   int root) {
    return mllm_broadcast_stream(comm, buffer, element_count, dtype, root, 0);
}

int mllm_broadcast_stream(mllm_comm_group_t *comm,
                          void *buffer,
                          size_t element_count,
                          mllm_dtype_t dtype,
                          int root,
                          cudaStream_t stream) {
    MLLM_CHECK(comm != NULL, "comm is NULL");
    MLLM_CHECK(buffer != NULL, "buffer is NULL");

    ncclDataType_t nccl_dtype = mllm_dtype_to_nccl(dtype);
    ncclResult_t rc = ncclBroadcast(buffer, buffer, element_count,
                                     nccl_dtype, root, comm->nccl_comm, stream);
    if (rc != ncclSuccess) {
        MLLM_LOG_ERROR("ncclBroadcast failed: %s", ncclGetErrorString(rc));
        return MLLM_ERR_NCCL;
    }
    return MLLM_OK;
}

int mllm_reduce(mllm_comm_group_t *comm,
                void *input_buf,
                void *output_buf,
                size_t element_count,
                mllm_dtype_t dtype,
                mllm_reduce_op_t op,
                int root) {
    return mllm_reduce_stream(comm, input_buf, output_buf, element_count,
                              dtype, op, root, 0);
}

int mllm_reduce_stream(mllm_comm_group_t *comm,
                       void *input_buf,
                       void *output_buf,
                       size_t element_count,
                       mllm_dtype_t dtype,
                       mllm_reduce_op_t op,
                       int root,
                       cudaStream_t stream) {
    MLLM_CHECK(comm != NULL, "comm is NULL");
    MLLM_CHECK(input_buf != NULL, "input_buf is NULL");
    MLLM_CHECK(output_buf != NULL, "output_buf is NULL");

    ncclDataType_t nccl_dtype = mllm_dtype_to_nccl(dtype);
    ncclRedOp_t nccl_op = mllm_reduce_op_to_nccl(op);
    ncclResult_t rc = ncclReduce(input_buf, output_buf, element_count,
                                  nccl_dtype, nccl_op, root,
                                  comm->nccl_comm, stream);
    if (rc != ncclSuccess) {
        MLLM_LOG_ERROR("ncclReduce failed: %s", ncclGetErrorString(rc));
        return MLLM_ERR_NCCL;
    }
    return MLLM_OK;
}

int mllm_send(mllm_comm_group_t *comm,
              const void *buf,
              size_t element_count,
              mllm_dtype_t dtype,
              int dest) {
    MLLM_CHECK(comm != NULL, "comm is NULL");
    MLLM_CHECK(buf != NULL, "buf is NULL");

    ncclDataType_t nccl_dtype = mllm_dtype_to_nccl(dtype);
    ncclResult_t rc = ncclSend(buf, element_count, nccl_dtype, dest,
                                comm->nccl_comm, 0);
    if (rc != ncclSuccess) {
        MLLM_LOG_ERROR("ncclSend failed: %s", ncclGetErrorString(rc));
        return MLLM_ERR_NCCL;
    }
    return MLLM_OK;
}

int mllm_recv(mllm_comm_group_t *comm,
              void *buf,
              size_t element_count,
              mllm_dtype_t dtype,
              int src) {
    MLLM_CHECK(comm != NULL, "comm is NULL");
    MLLM_CHECK(buf != NULL, "buf is NULL");

    ncclDataType_t nccl_dtype = mllm_dtype_to_nccl(dtype);
    ncclResult_t rc = ncclRecv(buf, element_count, nccl_dtype, src,
                                comm->nccl_comm, 0);
    if (rc != ncclSuccess) {
        MLLM_LOG_ERROR("ncclRecv failed: %s", ncclGetErrorString(rc));
        return MLLM_ERR_NCCL;
    }
    return MLLM_OK;
}

int mllm_stream_synchronize(mllm_comm_group_t *comm, cudaStream_t stream) {
    if (comm == NULL || stream == NULL) {
        return MLLM_ERR_INVALID_INPUT;
    }
    return MLLM_CUDA_CHECK(cudaStreamSynchronize(stream));
}

int mllm_create_comm_stream(mllm_comm_group_t *comm, cudaStream_t *out_stream) {
    MLLM_CHECK(comm != NULL, "comm is NULL");
    MLLM_CHECK(out_stream != NULL, "out_stream is NULL");

    return mllm_create_stream("comm_stream", out_stream);
}

void mllm_destroy_stream(cudaStream_t stream) {
    if (stream == NULL) return;
    cudaStreamDestroy(stream);
}

const char* mllm_nccl_version(void) {
    return ncclGetVersionString();
}

bool mllm_nccl_available(void) {
    // Check if NCCL is available on this system
    return true; // placeholder
}
