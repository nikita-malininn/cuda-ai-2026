cmake_minimum_required(VERSION 3.10)

project(6_softmax_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)


find_package(CUDAToolkit REQUIRED)

add_executable(softmax_cuda main.cpp softmax_cuda.cu)
target_link_libraries(softmax_cuda PRIVATE)