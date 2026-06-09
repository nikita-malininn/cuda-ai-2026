#include "block_gemm_cuda.h"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iostream>

#define BLOCK_SIZE 16

__global__ void blockMultKernel(const float* A, const float* B, float* C, int N)
{
    __shared__ float blkA[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ float blkB[BLOCK_SIZE*BLOCK_SIZE];

    int local_row = threadIdx.y;
    int local_col = threadIdx.x;

    int global_row = blockIdx.y * blockDim.y + threadIdx.y;
    int global_col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int block = 0; block < gridDim.x; ++block)
    {
        blkA[local_row*BLOCK_SIZE+local_col] = A[global_row*N+(block*BLOCK_SIZE+local_col)];
        blkB[local_row*BLOCK_SIZE+local_col] = B[(block*BLOCK_SIZE+local_row)*N+global_col];

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            sum += blkA[local_row*BLOCK_SIZE+i] * blkB[i*BLOCK_SIZE+local_col];
        }

        __syncthreads();
    }

    C[global_row*N+global_col] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    const int dataSize = n * n;
    const size_t bytes = dataSize * sizeof(float);

    const float* in_a = a.data();
    const float* in_b = b.data();

    std::vector<float> output(dataSize);
    float *out_c = output.data();

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, in_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, in_b, bytes, cudaMemcpyHostToDevice);

    //dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    //dim3 dimGrid(size, size);

    dim3 block_count(n/BLOCK_SIZE, n/BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);

    blockMultKernel<<<block_count, block_size>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(out_c, d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return output;
}

// int main(int argc, char *argv[])
// {
//     int size = 256;

//     std::vector<float> resMult;
//     std::vector<float> in_a(size*size, 0), in_b(size*size, 0);

//     srand(1234);
//     for (int i = 0; i < size * size; ++i) {
//         in_a[i] = static_cast<float>(rand()) / RAND_MAX;
//         in_b[i] = static_cast<float>(rand()) / RAND_MAX;
//     }

//     resMult = BlockGemmCUDA(in_a, in_b, size);

//     // Performance Measuring
//     std::vector<double> time_list;
//     for (int i = 0; i < 20; ++i) {
//         auto start = std::chrono::high_resolution_clock::now();
//         resMult = BlockGemmCUDA(in_a, in_b, size);
//         auto end = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> duration = end - start;
//         time_list.push_back(duration.count());
//     }
//     double time = *std::min_element(time_list.begin(), time_list.end());

//     std::cout << time << std::endl;

//     std::vector<float> resCPUMult(size*size, 0);
//     int N = size;
//     for (int i = 0; i < N; ++i)
//     {
//         for (int j = 0; j < N; ++j)
//         {
//             float res = 0.0f;
//             for (int k = 0; k < N; ++k)
//             {
//                 res += in_a[i * N + k] * in_b[k * N + j];
//             }
//             resCPUMult[i * N + j] = res;
//         }
//     }

//     double maxErr = 0.0;
//     for (int i = 0; i < N * N; ++i)
//     {
//         double err = fabs(resMult[i] - resCPUMult[i]);
//         if (err > maxErr) maxErr = err;
//     }
//     printf("Maximum absolute error: %e\n", maxErr);

//     return 0;
// }
