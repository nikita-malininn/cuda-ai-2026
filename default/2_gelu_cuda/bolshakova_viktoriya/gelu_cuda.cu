#include "gelu_cuda.h"

__host__ __forceinline__ int ceilDiv(int a, int b)
{
    return (a + b - 1) / b;
}

__device__ __forceinline__ float geluScalar(float x)
{
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float COEFF = 0.044715f;

    float x2 = x * x;
    float x3 = x2 * x;

    float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    if (inner > 10.0f)
    {
        inner = 10.0f;
    }
    if (inner < -10.0f)
    {
        inner = -10.0f;
    }

    float exp = __expf(2.0f * inner);
    float tanh = (exp - 1.0f) / (exp + 1.0f);

    return 0.5f * x * (1.0f + tanh);
}

__global__ void GeluKernel(const float *__restrict__ input, float *__restrict__ output, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        output[i] = geluScalar(input[i]);
    }
}

std::vector<float> GeluCUDA(const std::vector<float> &input)
{
    const int size = static_cast<int>(input.size());
    size_t bytes = size * sizeof(float);

    const float *inputData = input.data();
    std::vector<float> output(size);

    float* inputBuffer = nullptr;
    float* outputBuffer = nullptr;
    cudaMalloc(&inputBuffer, bytes);
    cudaMalloc(&outputBuffer, bytes);

    cudaMemcpy(inputBuffer, inputData, bytes, cudaMemcpyHostToDevice);

    const int blockSize = 256;
    int numBlocks = ceilDiv(size, blockSize);
    GeluKernel<<<numBlocks, blockSize>>>(inputBuffer, outputBuffer, size);
    cudaDeviceSynchronize();
    cudaMemcpy(output.data(), outputBuffer, bytes, cudaMemcpyDeviceToHost);

    cudaFree(inputBuffer);
    cudaFree(outputBuffer);

    return output;
}
