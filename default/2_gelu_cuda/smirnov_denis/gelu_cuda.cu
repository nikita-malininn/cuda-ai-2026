#include <cuda/cmath>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

#include "gelu_cuda.h"

__global__ void vecGelu(const float* X, float* Y, size_t n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        float x = X[idx];
        float inner = 0.79788456f * x * (1.f + 0.044715f * x * x);
        Y[idx] = 0.5f * x * (1.f + tanhf(inner));
    }
}

__global__ void vecGelu4(const float* X, float* Y, size_t n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        const float4* X4 = reinterpret_cast<const float4*>(X);
        float4* Y4 = reinterpret_cast<float4*>(Y);
        float4 x = X4[idx], y;
        y.x = 0.5f * x.x * (1.f + tanhf(0.79788456f * x.x * (1.f + 0.044715f * x.x * x.x)));
        y.y = 0.5f * x.y * (1.f + tanhf(0.79788456f * x.y * (1.f + 0.044715f * x.y * x.y)));
        y.z = 0.5f * x.z * (1.f + tanhf(0.79788456f * x.z * (1.f + 0.044715f * x.z * x.z)));
        y.w = 0.5f * x.w * (1.f + tanhf(0.79788456f * x.w * (1.f + 0.044715f * x.w * x.w)));
        Y4[idx] = y;
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();

    float* X = nullptr;
    float* Y = nullptr;

    cudaMalloc(&X, n * sizeof(float));
    cudaMalloc(&Y, n * sizeof(float));
    
    cudaMemcpy(X, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    size_t threads = 256;
    if (n % 4 == 0) {
        size_t blocks = (n / 4 + threads - 1) / threads;
        vecGelu4<<<blocks, threads>>>(X, Y, n);
    } else {
        size_t blocks = (n + threads - 1) / threads;
        vecGelu<<<blocks, threads>>>(X, Y, n);
    }

    std::vector<float> output(n);
    cudaMemcpy(output.data(), Y, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(X);
    cudaFree(Y);
    
    return output;
}

#if 0
std::vector<float> GeluRef(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);

    const float* inptr = input.data();
    float* outptr = output.data();

    for (size_t i = 0; i < n; i++) {
        float x = input[i];
        float y = 0.5f*x*(1 + std::tanh(std::sqrt(2.f/M_PI)*x*(1.f + 0.044715f*x*x)));
        output[i] = y;
    }

    return output;
}

int main() {
    size_t n = 134217728u;
    std::vector<float> x(n);
    for (size_t i = 0; i < n; i++) {
        x[i] = ((float)rand()/RAND_MAX)*20.f - 10.f;
    }

    // Warming-up
    auto y = GeluCUDA(x);

    std::vector<float> yref = GeluRef(x);
    float err = 0.f;
    for (size_t i = 0; i < n; i++) {
        err = std::max(err, std::abs(y[i] - yref[i]));
    }
    printf("max absolute error = %.5g\n", err);
    
    // Performance Measuring
    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto y = GeluCUDA(x);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());
    printf("time = %.4f\n", time);

    return 0;
}
#endif
