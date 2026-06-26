#include "block_gemm_cuda.h"

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda/cmath>
#include <cmath>
#include <cstdlib>
#include <chrono>


#define TILE_DIM 16

__global__ void tiled_matrix_mul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N)
{
    __shared__ float tileA[TILE_DIM*TILE_DIM];
    __shared__ float tileB[TILE_DIM*TILE_DIM];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int iBlock = 0; iBlock < gridDim.x; ++iBlock)
    {
        tileA[ty*TILE_DIM+tx] = A[row*N+(iBlock*TILE_DIM+tx)];
        tileB[ty*TILE_DIM+tx] = B[(iBlock*TILE_DIM+ty)*N+col];

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_DIM; ++i)
        {
            sum += tileA[ty*TILE_DIM+i] * tileB[i*TILE_DIM+tx];
        }

        __syncthreads();
    }

    if (row < N && col < N)
    {
        C[row*N+col] = sum;
    }

     __syncthreads();
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n)
{
    size_t N = n * n;
    size_t mtxSize = N * sizeof(float);
    std::vector<float> c(N);

    size_t sharedMemBytes = 2 * TILE_DIM * TILE_DIM * sizeof(float);

    float *a_ptr = nullptr;
    float *b_ptr = nullptr;
    float *c_ptr = nullptr;

    cudaMalloc(&a_ptr, mtxSize);
    cudaMalloc(&b_ptr, mtxSize);
    cudaMalloc(&c_ptr, mtxSize);

    cudaMemcpy(a_ptr, a.data(), mtxSize, cudaMemcpyHostToDevice);
    cudaMemcpy(b_ptr, b.data(), mtxSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 blocksPerGrid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    tiled_matrix_mul_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>>(a_ptr, b_ptr, c_ptr, n);

    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), c_ptr, mtxSize, cudaMemcpyDeviceToHost);

    cudaFree(a_ptr);
    cudaFree(b_ptr);
    cudaFree(c_ptr);

    return c;
}
