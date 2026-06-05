
project(test_block_gemm_cuda LANGUAGES CXX CUDA)

cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(test_block_gemm_cuda main.cpp block_gemm_cuda.cu block_gemm_cuda.h)
target_include_directories(test_block_gemm_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
