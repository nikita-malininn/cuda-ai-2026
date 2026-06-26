#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <thread>

namespace {

constexpr int kBlock = 256;
constexpr int kWarp = 32;

struct DataStore {
    float* gpu_data = nullptr;
    size_t capacity = 0;

    void ensure(size_t bytes) {
        if (bytes > capacity) {
            if (gpu_data) cudaFree(gpu_data);
            cudaMalloc(&gpu_data, bytes);
            capacity = bytes;
        }
    }

    ~DataStore() {
        if (gpu_data) cudaFree(gpu_data);
    }
};

static DataStore g_storage;

}

__device__ __forceinline__ float reduce_max(float val) {
    for (int s = kWarp / 2; s > 0; s >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, s));
    return val;
}

__device__ __forceinline__ float reduce_sum(float val) {
    for (int s = kWarp / 2; s > 0; s >>= 1)
        val += __shfl_down_sync(0xffffffff, val, s);
    return val;
}

__global__ void softmax_kernel(float* data, int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;

    int t = threadIdx.x;
    int lane = t & (kWarp - 1);
    int wid = t / kWarp;

    __shared__ float shmem[kBlock];
    float* row = data + r * cols;

    float lmax = -INFINITY;
    for (int c = t; c < cols; c += kBlock)
        lmax = fmaxf(lmax, row[c]);

    float wmax = reduce_max(lmax);
    if (lane == 0) shmem[wid] = wmax;
    __syncthreads();

    float gmax = (t < (kBlock / kWarp)) ? shmem[lane] : -INFINITY;
    if (wid == 0) gmax = reduce_max(gmax);
    if (t == 0) shmem[0] = gmax;
    __syncthreads();

    gmax = shmem[0];

    float lsum = 0.0f;
    for (int c = t; c < cols; c += kBlock) {
        float ev = expf(row[c] - gmax);
        row[c] = ev;
        lsum += ev;
    }

    float wsum = reduce_sum(lsum);
    if (lane == 0) shmem[wid] = wsum;
    __syncthreads();

    float gsum = (t < (kBlock / kWarp)) ? shmem[lane] : 0.0f;
    if (wid == 0) gsum = reduce_sum(gsum);
    if (t == 0) shmem[0] = __fdividef(1.0f, gsum);
    __syncthreads();

    float inv = shmem[0];
    for (int c = t; c < cols; c += kBlock)
        row[c] *= inv;
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    const size_t nelem = input.size();
    std::vector<float> output;

    std::thread alloc([&]() { output.resize(nelem); });

    const size_t cols = nelem / row_count;
    const size_t bytes = nelem * sizeof(float);

    g_storage.ensure(bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(g_storage.gpu_data, input.data(), bytes, cudaMemcpyHostToDevice, stream);

    softmax_kernel<<<row_count, kBlock, 0, stream>>>(g_storage.gpu_data, row_count, cols);

    alloc.join();
    cudaMemcpyAsync(output.data(), g_storage.gpu_data, bytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return output;
}
