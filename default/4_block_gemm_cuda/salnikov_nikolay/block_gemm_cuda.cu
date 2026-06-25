#include "block_gemm_cuda.h"

#define BLOCK_SIZE 16

__global__ void BlockGemmKernel(const float* A, const float* B, float* C, int N) {

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float cValue = 0.0f;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int numBlocks = N / BLOCK_SIZE;
    for (int k = 0; k < numBlocks; ++k)
    {
        if (row < N && (k * BLOCK_SIZE + tx) < N)
        {
            As[ty][tx] = A[row * N + (k * BLOCK_SIZE + tx)];
        }
        else
        {
            As[ty][tx] = 0.0f;
        }

        if ((k * BLOCK_SIZE + ty) < N && col < N)
        {
            Bs[ty][tx] = B[(k * BLOCK_SIZE + ty) * N + col];
        }
        else
        {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            cValue += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N)
    {
        C[row * N + col] = cValue;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    std::vector<float> c(n * n);

    const float* aptr = a.data();
    const float* bptr = b.data();
    float* cptr = c.data();

    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;

    int size = n * n * sizeof(float);
    cudaMalloc(&A, size);
    cudaMalloc(&B, size);
    cudaMalloc(&C, size);
    
    cudaMemcpy(A, aptr, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B, bptr, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    BlockGemmKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, n);

    cudaMemcpy(cptr, C, size, cudaMemcpyDeviceToHost);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    
    return c;
}