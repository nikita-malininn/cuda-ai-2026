#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <vector>

constexpr int BLOCK_SIZE = 256;

struct Context {
    float* dInput = nullptr;
    float* dOutput = nullptr;
    size_t allocated_bytes = 0;

    void EnsureCapacity(size_t bytes) {
        if (bytes <= allocated_bytes)
            return;

        if (dInput) cudaFree(dInput);
        if (dOutput) cudaFree(dOutput);

        cudaMalloc(&dInput, bytes);
        cudaMalloc(&dOutput, bytes);

        allocated_bytes = bytes;
    }

    ~Context() {
        if (dInput) cudaFree(dInput);
        if (dOutput) cudaFree(dOutput);
    }
};

static Context ctx;

__global__ void SoftmaxKernel(const float* input,
                              float* output,
                              int row_size) {
    __shared__ float sdata[BLOCK_SIZE];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = row * row_size;

    float local_max = -FLT_MAX;

    for (int i = tid; i < row_size; i += BLOCK_SIZE)
        local_max = fmaxf(local_max, input[base + i]);

    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }

    float row_max = sdata[0];

    float local_sum = 0.0f;

    for (int i = tid; i < row_size; i += BLOCK_SIZE) {
        float v = __expf(input[base + i] - row_max);
        output[base + i] = v;
        local_sum += v;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    float inv_sum = 1.0f / sdata[0];

    for (int i = tid; i < row_size; i += BLOCK_SIZE)
        output[base + i] *= inv_sum;
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input,
                               int row_count) {
    const int row_size = static_cast<int>(input.size()) / row_count;
    const size_t bytes = input.size() * sizeof(float);

    ctx.EnsureCapacity(bytes);
    cudaMemcpy(ctx.dInput, input.data(), bytes, cudaMemcpyHostToDevice);

    SoftmaxKernel<<<row_count, BLOCK_SIZE>>>( ctx.dInput, ctx.dOutput, row_size);

    std::vector<float> output(input.size());
    cudaMemcpy(output.data(), ctx.dOutput, bytes, cudaMemcpyDeviceToHost);

    return output;
}