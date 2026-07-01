#!/usr/bin/env bash
# Shared toolchain environment for the entropy C/CUDA trainer.
# Source this before building or running.
export ENTROPY_ROOT="/project/inniang/entropy"
export CUDAENV="${ENTROPY_ROOT}/.cudaenv"
export NVCC="${CUDAENV}/bin/nvcc"

# Self-consistent CUDA 12.8 toolkit (nvcc, cuBLAS/cuBLASLt, cudart, NCCL 2.30).
export CUDA_INC="${CUDAENV}/targets/x86_64-linux/include"
export CUDA_INC2="${CUDAENV}/include"
export CUDA_LIB="${CUDAENV}/targets/x86_64-linux/lib"
export CUDA_LIB2="${CUDAENV}/lib"

# cuDNN 9 (fused flash attention) from the pip wheel + its C++ frontend headers.
export VENV_SP="/project/inniang/.venv/lib/python3.13/site-packages"
export CUDNN_INC="${VENV_SP}/nvidia/cudnn/include"
export CUDNN_LIB="${VENV_SP}/nvidia/cudnn/lib"
export CUDNN_FE_INC="${VENV_SP}/include"

# Runtime: our toolkit libs first, then cuDNN, then the system driver (libcuda.so).
export LD_LIBRARY_PATH="${CUDA_LIB}:${CUDA_LIB2}:${CUDNN_LIB}:/usr/lib64:${LD_LIBRARY_PATH:-}"

# Default GPU architecture: RTX 6000 Ada Generation == sm_89.
export GPU_ARCH="${GPU_ARCH:-89}"
