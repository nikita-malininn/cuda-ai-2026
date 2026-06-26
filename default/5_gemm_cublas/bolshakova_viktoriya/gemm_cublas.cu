#include "gemm_cublas.h"

#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float> &a,
                              const std::vector<float> &b,
                              int n)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    const size_t totalElements = n * n;
    std::vector<float> c(totalElements);

    const size_t bytes = totalElements * sizeof(float);

    float* aBuffer = nullptr;
    float* bBuffer = nullptr;
    float* cBuffer = nullptr;
           
    cudaMalloc(&aBuffer, bytes);
    cudaMalloc(&bBuffer, bytes);
    cudaMalloc(&cBuffer, bytes);

    cudaMemcpyAsync(aBuffer, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(bBuffer, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    const float alpha = 1.0;
    const float beta = 0.0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,  &alpha, bBuffer, n, aBuffer, n, &beta, cBuffer, n);

    cudaMemcpyAsync(c.data(), cBuffer, bytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cudaFree(aBuffer);
    cudaFree(bBuffer);
    cudaFree(cBuffer);

    cublasDestroy(handle);
    cudaStreamDestroy(stream);

    return c;
}