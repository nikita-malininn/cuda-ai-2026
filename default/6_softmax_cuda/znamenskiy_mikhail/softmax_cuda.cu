#include "softmax_cuda.h"

#include <float.h>

#include <cuda/cmath>

constexpr int BLOCK_SIZE = 32;

__global__ void SoftmaxCUDAImpl(const float *input, float *output, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    bool general_calc = tid == 0;

    float local_max = -FLT_MAX;
    for (int col = tid; col < cols; col += BLOCK_SIZE) {
        local_max = cuda::std::fmax(input[row * cols + col], local_max);
    }

    __shared__ float local_maxes[BLOCK_SIZE];
    __shared__ float max_in_row;
    local_maxes[tid] = local_max;
    __syncthreads();

    if (general_calc) {
        max_in_row = -FLT_MAX;
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            max_in_row = cuda::std::fmax(local_maxes[i], max_in_row);
        }
    }

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += BLOCK_SIZE) {
        local_sum += cuda::std::expf(input[row * cols + col] - max_in_row);
    }

    __shared__ float local_sums[BLOCK_SIZE];
    __shared__ float sum_in_row;
    local_sums[tid] = local_sum;
    __syncthreads();

    if (general_calc) {
        sum_in_row = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum_in_row += local_sums[i];
        }
    }

    for (int col = tid; col < cols; col += BLOCK_SIZE) {
        output[row * cols + col] = cuda::std::expf(input[row * cols + col] - max_in_row) / sum_in_row;
        }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    const int mxSize = input.size();
    const int mxSizeBytes = mxSize * sizeof(float);
    const int col_count = mxSize / row_count;

    float* gpuBuffer = nullptr;
    // Allocating Cuda memory once
    cudaMalloc(&gpuBuffer, 2 * mxSizeBytes);
    cudaMemcpy(gpuBuffer, input.data(), mxSizeBytes, cudaMemcpyHostToDevice);

    SoftmaxCUDAImpl<<<row_count, BLOCK_SIZE>>>(gpuBuffer, gpuBuffer + mxSize, col_count);

    // Allocating result buffer while CUDA calculations are running
    std::vector<float> output(mxSize);
    float* outputData = output.data();

    cudaDeviceSynchronize();
    cudaMemcpy(outputData, gpuBuffer + mxSize, mxSizeBytes, cudaMemcpyDeviceToHost);
    cudaFree(gpuBuffer);

    return output;
}

