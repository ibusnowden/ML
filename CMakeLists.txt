cmake_minimum_required(VERSION 3.20)
project(entropy LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find NCCL
find_package(PkgConfig REQUIRED)
pkg_check_modules(NCCL REQUIRED nccl)

# Source directories
set(SRC_DIR ${CMAKE_SOURCE_DIR}/csrc/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/csrc/include)
set(KERNELS_DIR ${CMAKE_SOURCE_DIR}/csrc/kernels)

# Include directories
include_directories(${INCLUDE_DIR})
include_directories(${NCCL_INCLUDE_DIRS})

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xptxas -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

# Default architecture (RTX 6000 Ada), overridable via -DARCH=
if(NOT ARCH)
    set(ARCH "sm_89")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${ARCH}")

option(ENTROPY_BUILD_CSRC_PROTOTYPE "Build the older csrc prototype target" OFF)

set(CUDNN_INC "$ENV{CUDNN_INC}")
set(CUDNN_FE_INC "$ENV{CUDNN_FE_INC}")
set(CUDNN_LIB "$ENV{CUDNN_LIB}")

if(NOT CUDNN_INC OR NOT CUDNN_FE_INC OR NOT CUDNN_LIB)
    message(FATAL_ERROR
        "CUDNN_INC, CUDNN_FE_INC, and CUDNN_LIB must be set. "
        "Run `source scripts/env.sh` before configuring CMake.")
endif()

add_executable(gpt train/gpt.cu)
string(REGEX REPLACE "^sm_" "" ENTROPY_ARCH_NUM "${ARCH}")
target_compile_definitions(gpt PRIVATE USE_CUDNN ENTROPY_BUILD_SM=${ENTROPY_ARCH_NUM})
target_include_directories(gpt PRIVATE
    ${CMAKE_SOURCE_DIR}/train
    ${NCCL_INCLUDE_DIRS}
    ${CUDNN_INC}
    ${CUDNN_FE_INC}
)
target_link_directories(gpt PRIVATE
    ${NCCL_LIBRARY_DIRS}
    ${CUDNN_LIB}
)
target_link_libraries(gpt PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    ${NCCL_LIBRARIES}
    :libcudnn.so.9
    :libcudnn_graph.so.9
)

# All source files (implementation .cu files only)
set(ENTROPY_SOURCES
    ${SRC_DIR}/comm.cu
    ${SRC_DIR}/cuda_stream.cu
    ${SRC_DIR}/error.cu
    ${SRC_DIR}/layer_norm.cu
    ${SRC_DIR}/logging.cu
    ${SRC_DIR}/metrics.cu
    ${SRC_DIR}/nccl_wrapper.cu
    ${SRC_DIR}/tensor.cu
    ${SRC_DIR}/rms_norm.cu
    ${SRC_DIR}/model.cu
    ${SRC_DIR}/optimizer.cu
    ${SRC_DIR}/fsdp.cu
    ${SRC_DIR}/checkpoint.cu
    ${SRC_DIR}/data_loader.cu
    ${SRC_DIR}/main.cu
    ${KERNELS_DIR}/attention_kernels.cu
    ${KERNELS_DIR}/mlp_kernels.cu
    ${KERNELS_DIR}/moe_kernels.cu
    ${KERNELS_DIR}/activation_kernels.cu
    ${KERNELS_DIR}/gemm_kernels.cu
    ${KERNELS_DIR}/layer_norm_kernel.cu
    ${KERNELS_DIR}/vision_encoder.cu
)

if(ENTROPY_BUILD_CSRC_PROTOTYPE)
add_executable(entropy_train ${ENTROPY_SOURCES})

target_link_libraries(entropy_train PRIVATE
    ${NCCL_LIBRARIES}
    cudart
    pthread
    m
)

target_include_directories(entropy_train PRIVATE
    ${CMAKE_SOURCE_DIR}/csrc/include
    ${NCCL_INCLUDE_DIRS}
)

# Link flags
target_link_options(entropy_train PRIVATE
    -L${NCCL_LIBRARY_DIRS}
)
endif()

# Install
install(TARGETS gpt DESTINATION bin)

# Print config
message(STATUS "entropy build configuration:")
message(STATUS "  CUDA: ${CMAKE_CUDA_COMPILER}")
message(STATUS "  NCCL: ${NCCL_LIBRARIES}")
message(STATUS "  cuDNN lib: ${CUDNN_LIB}")
message(STATUS "  Arch: ${ARCH}")
message(STATUS "  Default target: gpt (train/gpt.cu)")
message(STATUS "  Build csrc prototype: ${ENTROPY_BUILD_CSRC_PROTOTYPE}")
