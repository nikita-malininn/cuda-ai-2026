#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <float.h>

// Anonymous namespace
namespace
{
    static const int s_WarpSize  = 32;
    static const int s_BlockSize = 256;
    static const int s_WarpMask  = 0xFFFFFFFF;
    static const int s_HalfWarp  = s_WarpSize / 2;

    __global__ void softmaxImpl(float* input, int numRows, int rowSize)
    {
        int threadId  = threadIdx.x;
        int warpId    = threadId / s_WarpSize;
        int laneId    = threadId % s_WarpSize;
        float* rowPtr = input + blockIdx.x * rowSize;

        float threadMax = -FLT_MAX;
        for (int i = threadId; i < rowSize; i += s_BlockSize)
            threadMax = fmaxf(threadMax, rowPtr[i]);

        for (int offset = s_HalfWarp; offset > 0; offset /= 2)
            threadMax = fmaxf(threadMax, __shfl_down_sync(s_WarpMask, threadMax, offset));

        __shared__ float warpLocalMaxes[s_WarpSize];
        if (laneId == 0)
            warpLocalMaxes[warpId] = threadMax;
        __syncthreads();

        float blockMax = (threadId < s_BlockSize / s_WarpSize) ? warpLocalMaxes[laneId] : -FLT_MAX;
        if (warpId == 0)
        {
            for (int offset = s_HalfWarp; offset > 0; offset /= 2)
                blockMax = fmaxf(blockMax, __shfl_down_sync(s_WarpMask, blockMax, offset));
            if (threadId == 0)
                warpLocalMaxes[0] = blockMax;
        }
        __syncthreads();
        blockMax = warpLocalMaxes[0];

        float threadSum = 0.0f;
        for (int i = threadId; i < rowSize; i += s_BlockSize)
        {
            float expVal = expf(rowPtr[i] - blockMax);
            rowPtr[i] = expVal;
            threadSum += expVal;
        }

        for (int offset = s_HalfWarp; offset > 0; offset /= 2)
            threadSum += __shfl_down_sync(s_WarpMask, threadSum, offset);

        __shared__ float warpLocalSums[s_WarpSize];
        if (laneId == 0) 
            warpLocalSums[warpId] = threadSum;
        __syncthreads();

        float blockSum = (threadId < s_BlockSize / s_WarpSize) ? warpLocalSums[laneId] : 0.0f;
        if (warpId == 0)
        {
            for (int offset = s_HalfWarp; offset > 0; offset /= 2)
                blockSum += __shfl_down_sync(s_WarpMask, blockSum, offset);
            if (threadId == 0)
                warpLocalSums[0] = blockSum;
        }
        __syncthreads();
        blockSum = warpLocalSums[0];

        float normalizationFactor = 1.0f / blockSum;
        for (int i = threadId; i < rowSize; i += s_BlockSize)
            rowPtr[i] *= normalizationFactor;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int numRows)
{
    // Place your implementation here
    const int dataSize = input.size();
    const int rowSize  = dataSize / numRows;
    
    std::vector<float> output(dataSize);
    
    float* deviceData  = nullptr;
    cudaMalloc(&deviceData, dataSize * sizeof(float));
    cudaMemcpyAsync(deviceData, input.data(), dataSize * sizeof(float), cudaMemcpyHostToDevice);
    softmaxImpl<<<numRows, s_BlockSize>>>(deviceData, numRows, rowSize);
  
    cudaMemcpyAsync(output.data(), deviceData, dataSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(0);
    cudaFree(deviceData);

    return output;
}
