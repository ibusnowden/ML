cmake_minimum_required(VERSION 3.20)
project(mmllm LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# Default architecture (H100), overridable via -DARCH=
if(NOT ARCH)
    set(ARCH "sm_90")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${ARCH}")

# All source files (implementation .cu files only)
set(MMLLM_SOURCES
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

# Build the training binary
add_executable(mmllm_train ${MMLLM_SOURCES})

target_link_libraries(mmllm_train PRIVATE
    ${NCCL_LIBRARIES}
    cudart
    pthread
    m
)

target_include_directories(mmllm_train PRIVATE
    ${CMAKE_SOURCE_DIR}/csrc/include
    ${NCCL_INCLUDE_DIRS}
)

# Link flags
target_link_options(mmllm_train PRIVATE
    -L${NCCL_LIBRARY_DIRS}
)

# Install
install(TARGETS mmllm_train DESTINATION bin)

# Print config
message(STATUS "mmllm build configuration:")
message(STATUS "  CUDA: ${CMAKE_CUDA_COMPILER}")
message(STATUS "  NCCL: ${NCCL_LIBRARIES}")
message(STATUS "  Arch: ${ARCH}")
message(STATUS "  Sources: ${MMLLM_SOURCES}")
