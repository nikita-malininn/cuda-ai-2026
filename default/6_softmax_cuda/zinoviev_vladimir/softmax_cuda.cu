#include <algorithm>
#include <chrono>
#include <vector>
#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include "softmax_cuda.h"

#define BLOCK_SIZE 256

__global__ void SoftmaxCUDAKernel(const float* input, float* output, int col_size) {
    __shared__ float reduce[BLOCK_SIZE];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    reduce[tid] = -__FLT_MAX__;
    for (int i = tid; i < col_size; i += BLOCK_SIZE) {
        reduce[tid] = fmaxf(input[bid * col_size + i], reduce[tid]);
    }
    __syncthreads();
    for ( int s = BLOCK_SIZE / 2; s > 32; s >>= 1 ) {
        if ( tid < s )
            reduce[tid] = fmaxf(reduce[tid + s], reduce[tid]);
        __syncthreads();
    }
    if ( tid < 32 ) {
        reduce[tid] = fmaxf(reduce[tid + 32], reduce[tid]); __syncwarp();
        reduce[tid] = fmaxf(reduce[tid + 16], reduce[tid]); __syncwarp();
        reduce[tid] = fmaxf(reduce[tid + 8], reduce[tid]); __syncwarp();
        reduce[tid] = fmaxf(reduce[tid + 4], reduce[tid]); __syncwarp();
        reduce[tid] = fmaxf(reduce[tid + 2], reduce[tid]); __syncwarp();
        reduce[tid] = fmaxf(reduce[tid + 1], reduce[tid]); __syncwarp();
    }
    __syncthreads();
    float row_max = reduce[0];
    __syncthreads();

    reduce[tid] = 0.f;
    for (int i = tid; i < col_size; i += BLOCK_SIZE) {
        reduce[tid] += __expf(input[bid * col_size + i] - row_max);
    }
    __syncthreads();
    for ( int s = BLOCK_SIZE / 2; s > 32; s >>= 1 ) {
        if ( tid < s )
            reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    if ( tid < 32 ) {
        reduce[tid] += reduce[tid + 32]; __syncwarp();
        reduce[tid] += reduce[tid + 16]; __syncwarp();
        reduce[tid] += reduce[tid + 8]; __syncwarp();
        reduce[tid] += reduce[tid + 4]; __syncwarp();
        reduce[tid] += reduce[tid + 2]; __syncwarp();
        reduce[tid] += reduce[tid + 1]; __syncwarp();
    }
    __syncthreads();
    float row_sum = reduce[0];
    for (int i = tid; i < col_size; i += BLOCK_SIZE) {
        int idx = bid * col_size + i;
        output[idx] = __expf(input[idx] - row_max) / row_sum;
    }
}


class SoftmaxCUDAHandler {
public:
    SoftmaxCUDAHandler() : d_in(nullptr), d_out(nullptr), memSizeLast(0) {
        cudaStreamCreate(&stream);
    }

    std::vector<float>& execute(const std::vector<float>& input, int row_size) {
        const int col_size = input.size() / row_size;
        const size_t memSize = input.size() * sizeof(float);
        if (memSize != memSizeLast) {
            if (d_in) {
                cudaFree(d_in);
                cudaFree(d_out);
            }
            cudaMalloc(&d_in, memSize);
            cudaMalloc(&d_out, memSize);
            output = std::vector<float>(input.size());
            memSizeLast = memSize;
        }

        cudaMemcpyAsync(this->d_in, input.data(), memSize, cudaMemcpyHostToDevice, stream);
        SoftmaxCUDAKernel<<<row_size, BLOCK_SIZE>>>(d_in, d_out, col_size);
        cudaMemcpyAsync(output.data(), d_out, memSize, cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);
        return output;
    }

    ~SoftmaxCUDAHandler() {
        cudaFree(d_in);
        cudaFree(d_out);

        cudaStreamDestroy(stream);
    }
private:
    cudaStream_t stream;
    std::vector<float> output;
    float *d_in, *d_out;
    size_t memSizeLast;
};

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_size) {
    static SoftmaxCUDAHandler handler;
    return handler.execute(input, row_size);
}
