#include "block_gemm_cuda.h"

int getBlockSize(const int n)
{
    if (n <= 256)
    {
        return 16;
    }
    return 32;
}

template <int BLOCK_SIZE>
__global__ void blockGemmExecute(const float *a, const float *b, float *c, int n)
{
    __shared__ float sharedBlockA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float sharedBlockB[BLOCK_SIZE * BLOCK_SIZE];

    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    int inCol = threadIdx.x;
    int inRow = threadIdx.y;

    int numBlocks = gridDim.x;

    float sum = 0.0f;
    for (int block = 0; block < numBlocks; ++block)
    {
        sharedBlockA[inRow * BLOCK_SIZE + inCol] = a[row * n + block * BLOCK_SIZE + inCol];
        sharedBlockB[inRow * BLOCK_SIZE + inCol] = b[(block * BLOCK_SIZE + inRow) * n + col];
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            sum += sharedBlockA[inRow * BLOCK_SIZE + k] * sharedBlockB[k * BLOCK_SIZE + inCol];
        }

        __syncthreads();
    }
    c[row * n + col] = sum;
}

template <int BLOCK_SIZE>
void blockGemmWrapper(const float *a, const float *b, float *c, int n,
               dim3 grid, dim3 block)
{
    blockGemmExecute<BLOCK_SIZE><<<grid, block>>>(a, b, c, n);
}

std::vector<float> BlockGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b,
                                 int n)
{
    const size_t totalElements = n * n;
    std::vector<float> c(totalElements);

    const size_t bytes = totalElements * sizeof(float);

    float *aBuffer = nullptr;
    float *bBuffer = nullptr;
    float *cBuffer = nullptr;

    cudaMalloc(&aBuffer, bytes);
    cudaMalloc(&bBuffer, bytes);
    cudaMalloc(&cBuffer, bytes);

    cudaMemcpy(aBuffer, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bBuffer, b.data(), bytes, cudaMemcpyHostToDevice);

    int blockSize = getBlockSize(n);
    int numBlocks = (n + blockSize - 1) / blockSize;

    dim3 block(blockSize, blockSize);
    dim3 grid(numBlocks, numBlocks);

    switch (blockSize) {
        case 16: {
            blockGemmWrapper<16>(aBuffer, bBuffer, cBuffer, n, grid, block); 
            break;
        }
        default: {
            blockGemmWrapper<32>(aBuffer, bBuffer, cBuffer, n, grid, block); 
            break;
        }
    }

    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), cBuffer, bytes, cudaMemcpyDeviceToHost);

    cudaFree(aBuffer);
    cudaFree(bBuffer);
    cudaFree(cBuffer);

    return c;
}


