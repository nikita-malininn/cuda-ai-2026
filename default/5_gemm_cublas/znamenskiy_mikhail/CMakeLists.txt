
project(test_gemm_cublas LANGUAGES CXX CUDA)

cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)

add_executable(test_gemm_cublas main.cpp gemm_cublas.cu gemm_cublas.h)
target_include_directories(test_gemm_cublas PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_gemm_cublas PRIVATE CUDA::cublas)
