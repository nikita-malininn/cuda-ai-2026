#include "block_gemm_cuda.h"

#include <cuda/cmath>

#define BLOCK_SIZE 16
#define BLOCK_SIZE_SQR BLOCK_SIZE * BLOCK_SIZE

__global__ void BlockGemmCUDAImpl(const float *a, const float *b, float *c, int n) {
    __shared__ float blockA[BLOCK_SIZE_SQR];
    __shared__ float blockB[BLOCK_SIZE_SQR];

    int xInBlock = threadIdx.x;
    int yInBlock = threadIdx.y;
    int numBlocks = gridDim.x;
    int blockSize = blockDim.x;
    int x = blockIdx.x * blockSize + xInBlock;
    int y = blockIdx.y * blockSize + yInBlock;

    float sum = 0.0f;
    for (int block = 0; block < numBlocks; ++block) {
        blockA[yInBlock * blockSize + xInBlock] = a[y * n + block * blockSize + xInBlock];
        blockB[yInBlock * blockSize + xInBlock] = b[(block * blockSize + yInBlock) * n + x];
        __syncthreads();

        for (int k = 0; k < blockSize; ++k) {
            sum += blockA[yInBlock * blockSize + k] * blockB[k * blockSize + xInBlock];
        }
        __syncthreads();
    }
    c[y * n + x] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
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

    constexpr int threadsInBlock = BLOCK_SIZE;
    int blocks = n / threadsInBlock;
    dim3 threadsXY(threadsInBlock, threadsInBlock);
    dim3 blocksXY(blocks, blocks);
    BlockGemmCUDAImpl<<<blocksXY, threadsXY>>>(gpuBuffer, gpuBuffer + mxSize, gpuBuffer + 2 * mxSize, n);

    // Allocating result buffer while CUDA calculations are running
    std::vector<float> c(mxSize);
    float* cData = c.data();

    cudaDeviceSynchronize();
    cudaMemcpy(cData, gpuBuffer + 2 * mxSize, mxSizeBytes, cudaMemcpyDeviceToHost);
    cudaFree(gpuBuffer);

    return c;
}
