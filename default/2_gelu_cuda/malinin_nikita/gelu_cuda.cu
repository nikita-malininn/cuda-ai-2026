#include "gelu_cuda.h"

#include <cuda/cmath>

__global__ void gelu(float* gpuBuffer, int bytesSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < bytesSize) {
        float x = gpuBuffer[idx];
        float inner = 0.79788456f * x * (1.f + 0.044715f * x * x);
        gpuBuffer[idx] = 0.5f * x * (1.f + tanhf(inner));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int inputSize = input.size();
    int bytesSize = inputSize * sizeof(float);
    const float* inputData = input.data();

    float* gpuBuffer = nullptr;
    cudaMalloc(&gpuBuffer, bytesSize);
    cudaMemcpy(gpuBuffer, inputData, bytesSize, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = cuda::ceil_div(inputSize, threads);
    gelu<<<blocks, threads>>>(gpuBuffer, inputSize);

    std::vector<float> result(inputSize);
    float* resultData = result.data();

    cudaDeviceSynchronize();
    cudaMemcpy(resultData, gpuBuffer, bytesSize, cudaMemcpyDeviceToHost);
    cudaFree(gpuBuffer);

    return result;
}
