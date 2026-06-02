#include "naive_gemm_cuda.h"

#include <cuda_runtime_api.h>
#include <cuda/cmath>
#include <vector>
#include <algorithm>

__global__ void kernelGemmSquare(const float * __restrict__  a, 
                                 const float * __restrict__  b, 
                                 float * __restrict__  c, int n) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < n && j < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[i * n + k] * b[k * n + j];
        }
        c[i * n + j] = sum;
    }
}

constexpr int blockdim_x = 32;
constexpr int blockdim_y = 32;

__global__ void kernelGemmSquareVec(const float *a, const float *b, float *c, int n) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i < n && j < n) {
        float sum = 0.0f;
        
        for (int k = 0; k < n; k += 4) {
            float4 a_vec = reinterpret_cast<const float4*>(&a[i * n + k])[0];
            float b0 = b[(k + 0) * n + j];
            float b1 = b[(k + 1) * n + j];
            float b2 = b[(k + 2) * n + j];
            float b3 = b[(k + 3) * n + j];
            
            sum += a_vec.x * b0 + a_vec.y * b1 + a_vec.z * b2 + a_vec.w * b3;
        }
        
        c[i * n + j] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {

    const int lenVec = n * n;
    const int size = lenVec * sizeof(float);

    float* a_dev = nullptr;
    cudaMalloc(&a_dev, size);
    float* b_dev = nullptr;
    cudaMalloc(&b_dev, size);

    float* c_dev = nullptr;
    cudaMalloc(&c_dev, size);

    const float* a_host = a.data();
    const float* b_host = b.data();
    
    cudaMemcpy(a_dev, a_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b_host, size, cudaMemcpyHostToDevice);

    dim3 threadsGrid(blockdim_x, blockdim_y);
    dim3 matrDim(n, n);

    dim3 blockGrid( 
        cuda::ceil_div(matrDim.x, threadsGrid.x),
        cuda::ceil_div(matrDim.y, threadsGrid.y)
    );

    kernelGemmSquareVec<<<blockGrid, threadsGrid>>>(a_dev, b_dev, c_dev, n);

    // kernelGemmSquare<<<blockGrid, threadsGrid>>>(a_dev, b_dev, c_dev, n);

    std::vector<float> c(lenVec); 

    cudaDeviceSynchronize();

    cudaMemcpyAsync(c.data(), c_dev, size, cudaMemcpyDeviceToHost);


    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);

    return c;
}
