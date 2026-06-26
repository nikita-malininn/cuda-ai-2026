#include "naive_gemm_cuda.h"

#include <vector>

__global__ void naiveGemmExecute(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const size_t totalElements = n * n;
    std::vector<float> c(totalElements);

    const size_t bytes = totalElements * sizeof(float);

    float* aBuffer = nullptr;
    float* bBuffer = nullptr;
    float* cBuffer = nullptr;
                      
    cudaMalloc(&aBuffer, bytes);
    cudaMalloc(&bBuffer, bytes);
    cudaMalloc(&cBuffer, bytes);

    cudaMemcpy(aBuffer, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bBuffer, b.data(), bytes, cudaMemcpyHostToDevice);

   dim3 block(16, 16);
   dim3 grid((n + block.x - 1) / block.x,
              (n + block.y - 1) / block.y);

    naiveGemmExecute<<<grid, block>>>(aBuffer, bBuffer, cBuffer, n);

    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), cBuffer, bytes, cudaMemcpyDeviceToHost);

    cudaFree(aBuffer);
    cudaFree(bBuffer);
    cudaFree(cBuffer);

    return c;
}