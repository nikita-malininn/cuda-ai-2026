#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__inline__ __device__ float warpReduceMax(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__inline__ __device__ float warpReduceSum(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__inline__ __device__ float blockReduceMax(float val, float* sharedMem)
{
    int laneId = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;

    val = warpReduceMax(val);
    if (laneId == 0)
    {
        sharedMem[warpId] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? sharedMem[laneId] : -INFINITY;
    if (warpId == 0)
    {
        val = warpReduceMax(val);
        if (threadIdx.x == 0)
        {
            sharedMem[0] = val;
        }
    }
    __syncthreads();
    return sharedMem[0];
}

__inline__ __device__ float blockReduceSum(float val, float* sharedMem)
{
    int laneId = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;

    val = warpReduceSum(val);
    if (laneId == 0)
    {
        sharedMem[warpId] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? sharedMem[laneId] : 0.0f;
    if (warpId == 0)
    {
        val = warpReduceSum(val);
        if (threadIdx.x == 0)
        {
            sharedMem[0] = val;
        }
    }
    __syncthreads();
    return sharedMem[0];
}

__global__ void SoftmaxCudaKernel(float* input, int rowCount, int rowSize)
{
    int rowIdx = blockIdx.x;
    if (rowIdx >= rowCount)
    {
        return;
    }

    int threadId = threadIdx.x;
    float* rowInputData = input + rowIdx * rowSize;

    __shared__ float sharedMem[WARP_SIZE];

    float threadMax = -INFINITY;
    for (int col = threadId; col < rowSize; col += blockDim.x)
    {
        threadMax = fmaxf(threadMax, rowInputData[col]);
    }

    float globalMax = blockReduceMax(threadMax, sharedMem);

    float threadSum = 0.0f;
    for (int col = threadId; col < rowSize; col += blockDim.x)
    {
        float expVal = expf(rowInputData[col] - globalMax);
        rowInputData[col] = expVal;
        threadSum += expVal;
    }

    float globalSum = blockReduceSum(threadSum, sharedMem);
    

    float invGlobalSum = 1.0f / globalSum;
    for (int col = threadId; col < rowSize; col += blockDim.x)
    {
        rowInputData[col] *= invGlobalSum;
    }
}


std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int rowCount)
{
    const int dataSize = input.size();
    
    std::vector<float> output(dataSize); 
    
    const int rowSize = dataSize / rowCount;
    const float* inputData = input.data();

    float* devInput = nullptr;
    cudaMalloc(&devInput, dataSize * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(devInput, inputData, dataSize * sizeof(float), cudaMemcpyHostToDevice, stream);

    SoftmaxCudaKernel<<<rowCount, BLOCK_SIZE, 0, stream>>>(devInput, rowCount, rowSize);

    cudaMemcpyAsync(output.data(), devInput, dataSize * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(devInput);

    return output;
}