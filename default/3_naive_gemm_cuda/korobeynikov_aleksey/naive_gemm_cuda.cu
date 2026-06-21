#include "naive_gemm_cuda.h"


__global__ void gemm_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[i * n + k] * b[k * n + j];
        }
        c[i * n + j] = sum;
    }
}


std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> result(a.size());

    float* a_gpu = nullptr;
    float* b_gpu = nullptr;
    float* result_gpu = nullptr;

    size_t size_in_bytes = a.size() * sizeof(float);

    cudaMalloc(&a_gpu, size_in_bytes);
    cudaMalloc(&b_gpu, size_in_bytes);
    cudaMalloc(&result_gpu, size_in_bytes);

    cudaMemcpy(a_gpu, a.data(), size_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b.data(), size_in_bytes, cudaMemcpyHostToDevice);

    dim3 threads_per_block(16, 16);
    dim3 blocks(
        (n + threads_per_block.x - 1) / threads_per_block.x,
        (n + threads_per_block.y - 1) / threads_per_block.y
    );

    gemm_kernel<<<blocks, threads_per_block>>>(a_gpu, b_gpu, result_gpu, n);

    cudaDeviceSynchronize();

    cudaMemcpy(result.data(), result_gpu, size_in_bytes, cudaMemcpyDeviceToHost);

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(result_gpu);

    return result;
}
