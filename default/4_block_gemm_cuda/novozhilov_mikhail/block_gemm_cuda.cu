#include "block_gemm_cuda.h"

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda/cmath>

__global__ void blockGemmKernel(const float *a, const float *b, float *c, int n, int blockSize)
{
    extern __shared__ float shared_buf[];
    float* bufA = shared_buf;
    float* bufB = &shared_buf[blockSize * blockSize];

    int row = blockIdx.y * blockSize + threadIdx.y;
    int col = blockIdx.x * blockSize + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (n + blockSize - 1) / blockSize;

    for (int t = 0; t < numTiles; ++t)
    {
        int aCol = t * blockSize + threadIdx.x;
        int bRow = t * blockSize + threadIdx.y;

        bufA[threadIdx.y * blockSize + threadIdx.x] = (row < n && aCol < n) ? a[row * n + aCol] : 0.0f;
        bufB[threadIdx.y * blockSize + threadIdx.x] = (bRow < n && col < n) ? b[bRow * n + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < blockSize; ++k)
        {
            sum += bufA[threadIdx.y * blockSize + k] * bufB[k * blockSize + threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n)
    {
        c[row * n + col] = sum;
    }
}


std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    constexpr int tileSize = 16;
    size_t sharedBufSize = 2 * tileSize * tileSize * sizeof(float);

    size_t size = n * n * sizeof(float);
    std::vector<float> c(n * n);

    float *g_a = nullptr;
    float *g_b = nullptr;
    float *g_c = nullptr;

    float *h_a = const_cast<float *>(a.data());
    float *h_b = const_cast<float *>(b.data());
    float *h_c = const_cast<float *>(c.data());

    cudaMalloc(&g_a, size);
    cudaMalloc(&g_b, size);
    cudaMalloc(&g_c, size);

    cudaMemcpy(g_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_b, h_b, size, cudaMemcpyHostToDevice);

    dim3 blockDims(tileSize, tileSize);

    int sizeGridDim = cuda::ceil_div(n, tileSize);
    dim3 gidDims(sizeGridDim, sizeGridDim);

    blockGemmKernel<<<gidDims, blockDims, sharedBufSize>>>(g_a, g_b, g_c, n, tileSize);

    cudaDeviceSynchronize();

    cudaMemcpy(h_c, g_c, size, cudaMemcpyDeviceToHost);

    cudaFree(g_a);
    cudaFree(g_b);
    cudaFree(g_c);

    return c;
}
