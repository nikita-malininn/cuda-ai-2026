#include "gemm_cublas.h"
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n)
{
    cublasHandle_t cublas;
    cublasCreate(&cublas);

    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;

    int size = n * n * sizeof(float);
    cudaMalloc(&A, size);
    cudaMalloc(&B, size);
    cudaMalloc(&C, size);

    cudaMemcpy(A, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(B, b.data(), size, cudaMemcpyHostToDevice);

    float alpha = 1.f;
    float beta = 0.f;
    cublasSgemm(cublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha, B, n, A, n,
                &beta, C, n);

    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), C, size, cudaMemcpyDeviceToHost);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    cublasDestroy(cublas);

    return c;
} //
