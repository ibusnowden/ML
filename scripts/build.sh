#!/usr/bin/env bash
# Build a single-TU CUDA program from train/ with the local CUDA 12.8 toolkit.
# Usage: scripts/build.sh <src.cu> <out_binary> [extra nvcc flags...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"

SRC="${1:?usage: build.sh <src.cu> <out> [flags...]}"
OUT="${2:?usage: build.sh <src.cu> <out> [flags...]}"
shift 2 || true

mkdir -p "$(dirname "${OUT}")"

set -x
"${NVCC}" -std=c++17 -O3 \
  -arch="sm_${GPU_ARCH}" \
  --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda \
  -Xcompiler -fPIC -lineinfo -DUSE_CUDNN -DENTROPY_BUILD_SM="${GPU_ARCH}" \
  -I"${CUDA_INC}" -I"${CUDA_INC2}" -I"${ENTROPY_ROOT}/train" \
  -I"${CUDNN_INC}" -I"${CUDNN_FE_INC}" \
  "${SRC}" -o "${OUT}" \
  -L"${CUDA_LIB}" -L"${CUDA_LIB2}" -L"${CUDNN_LIB}" \
  -lcudart -lcublas -lcublasLt -lnccl \
  -l:libcudnn.so.9 -l:libcudnn_graph.so.9 \
  "$@"
set +x
echo "[build] -> ${OUT}"
