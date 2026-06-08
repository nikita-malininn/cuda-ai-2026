#include "gemm_cublas.h"
#include <cublas_v2.h>

#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const int mxSize = a.size();
    const int mxSizeBytes = mxSize * sizeof(float);

    float* gpuBuffer = nullptr;
    // Allocating Cuda memory once
    cudaMalloc(&gpuBuffer, 3 * mxSizeBytes);
    cudaMemcpy(gpuBuffer, a.data(), mxSizeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuBuffer + mxSize, b.data(), mxSizeBytes, cudaMemcpyHostToDevice);
    cudaMemset(gpuBuffer + 2 * mxSize, 0, mxSizeBytes);

    cublasStatus_t result;
    cublasHandle_t handle;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    result = cublasCreate(&handle);
    if (result != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS create error: " << cublasGetStatusString(result) << std::endl;
    }
    result = cublasSgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,    // Atransp, Btransp
                           n, n, n,                     // m, n , k
                           &alpha,
                           gpuBuffer + mxSize, n,
                           gpuBuffer, n,
                           &beta,
                           gpuBuffer + 2 * mxSize, n);
    if (result != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS Sgemm error: " << cublasGetStatusString(result) << std::endl;
    }

    // Allocating result buffer
    std::vector<float> c(mxSize);
    float* cData = c.data();

    cudaMemcpy(cData, gpuBuffer + 2 * mxSize, mxSizeBytes, cudaMemcpyDeviceToHost);
    cudaFree(gpuBuffer);
    
    result = cublasDestroy(handle);
    if (result != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS destroy error: " << cublasGetStatusString(result) << std::endl;
    }

    return c;
}
