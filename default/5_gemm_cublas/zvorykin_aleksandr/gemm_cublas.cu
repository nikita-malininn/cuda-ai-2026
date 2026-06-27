#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n)
{
    // Place your implementation here
    static cublasHandle_t cublas_handle = nullptr;
    static cudaStream_t stream = nullptr;
    static float* device_a = nullptr;
    static float* device_b = nullptr;
    static float* device_c = nullptr;
    static int last_n = 0;
    static bool initialized = false;
  
    if (!initialized)
    {
      cublasCreate(&cublas_handle);
      cudaStreamCreate(&stream);
      cublasSetStream(cublas_handle, stream);
      cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
      initialized = true;
    }
  
    const size_t data_size = sizeof(float) * static_cast<size_t>(n) * n;
  
    if (n != last_n)
    {
      if (device_a)
      {
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);
      }
      cudaMalloc(&device_a, data_size);
      cudaMalloc(&device_b, data_size);
      cudaMalloc(&device_c, data_size);
      last_n = n;
    }
  
    const float alpha = 1.0f;
    const float beta = 0.0f;
  
    cudaMemcpy(device_a, a.data(), data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b.data(), data_size, cudaMemcpyHostToDevice);
  
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n,
                &alpha, device_a, n, device_b, n, &beta, device_c, n);
  
    std::vector<float> result(static_cast<size_t>(n) * n);
    cudaMemcpy(result.data(), device_c, data_size, cudaMemcpyDeviceToHost);
  
    return result;
}
