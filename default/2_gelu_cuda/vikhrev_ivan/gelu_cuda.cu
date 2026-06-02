#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

constexpr float SQRT_2_DIV_PI = 0.7978845608028654f;

inline float gelu(const float x) {
    return 0.5f * x *  (1.f + std::tanh(SQRT_2_DIV_PI * (x + 0.044715f * x * x * x)));
}

std::vector<float> Gelu(const std::vector<float>& input) {
    std::vector<float> result(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] = gelu(input[i]);
    }
    return result;
}


inline void check_error(cudaError_t ret_code, const std::string& message = "") {
    if (ret_code != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(ret_code));
    }
}

__device__  __forceinline__ float fast_tanh(float x) {
    return 1.f - (2.f / (1.f + std::exp(x * 2.f)));;
}

__global__ void gelu_kernel(const float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float x = input[i];
        float cx3 = 0.044715f * x * x * x;
        float tanh_val = fast_tanh(SQRT_2_DIV_PI * (x + cx3));
        output[i] = 0.5f * x *  (1.f + tanh_val);
    }
}

class CudaGeluRunner {
public:
    CudaGeluRunner() {}

    ~CudaGeluRunner() {
        cudaFree(d_input);
        cudaFree(d_output);
    }


    CudaGeluRunner(const CudaGeluRunner&) = delete;
    CudaGeluRunner& operator=(const CudaGeluRunner&) = delete;

    std::vector<float> run(const std::vector<float>& input) {
        if (input.empty()) {
            return {};
        }

        ensure_capacity(input.size());

        const int n = static_cast<int>(input.size());
        const std::size_t bytes = input.size() * sizeof(float);

        std::vector<float> result(input.size());

        check_error(cudaMemcpy(
            d_input,
            input.data(),
            bytes,
            cudaMemcpyHostToDevice
        ), "cudaMemcpy host to device failed");

        const int blocks = (n + threads_per_block - 1) / threads_per_block;
        gelu_kernel<<<blocks, threads_per_block>>>(d_input, d_output, n);

        check_error(cudaGetLastError(), "GELU kernel launch failed");
        check_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

        check_error(cudaMemcpy(
            result.data(),
            d_output,
            bytes,
            cudaMemcpyDeviceToHost
        ), "cudaMemcpy device to host failed");


        return result;
    }

private:
    void ensure_capacity(std::size_t requested_size) {
        if (requested_size <= capacity) {
            return;
        }

        check_error(cudaFree(d_input), "cudaFree d_input failed");
        check_error(cudaFree(d_output), "cudaFree d_output failed");

        const std::size_t num_bytes = requested_size * sizeof(float);

        check_error(cudaMalloc(&d_input, num_bytes), "cudaMalloc d_input failed");
        check_error(cudaMalloc(&d_output, num_bytes), "cudaMalloc d_output failed");

        capacity = requested_size;
    }

    const int threads_per_block = 256;

    std::size_t capacity = 0;

    float* d_input = nullptr;
    float* d_output = nullptr;
};

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    static CudaGeluRunner runner;
    return runner.run(input);
}

std::vector<float> GeluCUDANaive(const std::vector<float>& input) {
    int n = input.size();
    std::vector<float> result(input.size());

    float* d_input = nullptr;
    float* d_output = nullptr;

    check_error(cudaMalloc(&d_input, n * sizeof(float)));
    check_error(cudaMalloc(&d_output, n * sizeof(float)));

    check_error(cudaMemcpy(
        d_input,
        input.data(),
        n * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    gelu_kernel<<<blocks, threads_per_block>>>(d_input, d_output, n);

    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());

    cudaMemcpy(
        result.data(),
        d_output,
        n * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    check_error(cudaFree(d_input));
    check_error(cudaFree(d_output));

    return result;
}
