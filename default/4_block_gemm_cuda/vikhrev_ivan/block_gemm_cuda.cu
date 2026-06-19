#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

static constexpr int BLOCK_SIZE = 16;

std::vector<float> NaiveGemm(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Input vectors must have the same size");
    }

    if (a.size() != n * n) {
        throw std::invalid_argument("Input vector size must be equal to n * n");
    }

    std::vector<float> c(a.size(), 0.f);
    for (auto i = 0; i < n; ++i) {
        for (auto j = 0; j < n; ++j) {
            for (auto k = 0; k < n; ++k) {
                c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }

    return c;
}

inline void check_error(cudaError_t ret_code, const std::string& message = "") {
    if (ret_code != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(ret_code));
    }
}

template <int BLOCK_SIZE>
__global__ void gemm_kernel_generic(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int block_i = threadIdx.y;
    int block_j = threadIdx.x;

    __shared__ float block_a[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float block_b[BLOCK_SIZE * BLOCK_SIZE];

    float sum = 0.f;
    for (int block_n = 0; block_n < gridDim.x; ++block_n) {
        int a_col = block_n * BLOCK_SIZE + block_j;
        int b_row = block_n * BLOCK_SIZE + block_i;

        block_a[block_i * BLOCK_SIZE + block_j] = (i < n && a_col < n) ? a[i * n + a_col] : 0.f;
        block_b[block_i * BLOCK_SIZE + block_j] = (b_row < n && j < n) ? b[b_row * n + j] : 0.f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += block_a[block_i * BLOCK_SIZE + k] * block_b[k * BLOCK_SIZE + block_j];
        }
        __syncthreads();
    }

    if (i < n && j < n) {
        c[i * n + j] = sum;
    }
}

template <int BLOCK_SIZE>
__global__ void gemm_kernel_aligned(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int block_i = threadIdx.y;
    int block_j = threadIdx.x;

    __shared__ float block_a[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float block_b[BLOCK_SIZE * BLOCK_SIZE];

    float sum = 0.f;
    for (int block_n = 0; block_n < gridDim.x; ++block_n) {
        int a_col = block_n * BLOCK_SIZE + block_j;
        int b_row = block_n * BLOCK_SIZE + block_i;

        block_a[block_i * BLOCK_SIZE + block_j] = a[i * n + a_col];
        block_b[block_i * BLOCK_SIZE + block_j] = b[b_row * n + j];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += block_a[block_i * BLOCK_SIZE + k] * block_b[k * BLOCK_SIZE + block_j];
        }
        __syncthreads();
    }

    c[i * n + j] = sum;
}

class CudaGemmRunner {
public:
    CudaGemmRunner() {}

    ~CudaGemmRunner() {
        cudaFree(a_d_input);
        cudaFree(b_d_input);
        cudaFree(c_d_output);
    }


    CudaGemmRunner(const CudaGemmRunner&) = delete;
    CudaGemmRunner& operator=(const CudaGemmRunner&) = delete;

    std::vector<float> run(const std::vector<float>& a, const std::vector<float>& b, int n) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Input vectors must have the same size");
        }

        if (a.size() != n * n) {
            throw std::invalid_argument("Input vector size must be equal to n * n");
        }

        int elements_num = a.size();
        ensure_capacity(elements_num);

        std::vector<float> c(elements_num);

        check_error(cudaMemcpy(
            a_d_input,
            a.data(),
            elements_num * sizeof(float),
            cudaMemcpyHostToDevice
        ), "cudaMemcpy a_d_input failed");

        check_error(cudaMemcpy(
            b_d_input,
            b.data(),
            elements_num * sizeof(float),
            cudaMemcpyHostToDevice
        ), "cudaMemcpy b_d_input failed");

        const dim3 blocks((n + threads_per_block.x - 1) / threads_per_block.x,
            (n + threads_per_block.y - 1) / threads_per_block.y);

        bool aligned = n % BLOCK_SIZE == 0;
        if (aligned) {
            gemm_kernel_aligned<BLOCK_SIZE><<<blocks, threads_per_block>>>(a_d_input, b_d_input, c_d_output, n);
        } else {
            gemm_kernel_generic<BLOCK_SIZE><<<blocks, threads_per_block>>>(a_d_input, b_d_input, c_d_output, n);
        }

        check_error(cudaGetLastError(), "GELU kernel launch failed");
        check_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

        check_error(cudaMemcpy(
            c.data(),
            c_d_output,
            num_bytes,
            cudaMemcpyDeviceToHost
        ), "cudaMemcpy device to host failed");


        return c;
    }

private:
    void ensure_capacity(std::size_t requested_size) {
        if (requested_size <= capacity) {
            return;
        }

        check_error(cudaFree(a_d_input), "cudaFree a_d_input failed");
        check_error(cudaFree(b_d_input), "cudaFree b_d_input failed");
        check_error(cudaFree(c_d_output), "cudaFree c_d_output failed");

        num_bytes = requested_size * sizeof(float);

        check_error(cudaMalloc(&a_d_input, num_bytes), "cudaMalloc d_input failed");
        check_error(cudaMalloc(&b_d_input, num_bytes), "cudaMalloc d_input failed");
        check_error(cudaMalloc(&c_d_output, num_bytes), "cudaMalloc d_output failed");

        capacity = requested_size;
    }

    const dim3 threads_per_block = dim3(BLOCK_SIZE, BLOCK_SIZE);


    std::size_t capacity = 0;
    std::size_t num_bytes = 0;

    float* a_d_input = nullptr;
    float* b_d_input = nullptr;
    float* c_d_output = nullptr;
};

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    static CudaGemmRunner runner;
    return runner.run(a, b, n);
}

std::vector<float> BlockGemmCUDASimple(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Input vectors must have the same size");
    }

    if (a.size() != n * n) {
        throw std::invalid_argument("Input vector size must be equal to n * n");
    }

    int elements_num = a.size();
    std::vector<float> c(a.size());

    float* a_d_input = nullptr;
    float* b_d_input = nullptr;
    float* c_d_output = nullptr;

    check_error(cudaMalloc(&a_d_input, elements_num * sizeof(float)));
    check_error(cudaMalloc(&b_d_input, elements_num * sizeof(float)));
    check_error(cudaMalloc(&c_d_output, elements_num * sizeof(float)));

    check_error(cudaMemcpy(
        a_d_input,
        a.data(),
        elements_num * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    check_error(cudaMemcpy(
        b_d_input,
        b.data(),
        elements_num * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (n + threads_per_block.x - 1) / threads_per_block.x,
        (n + threads_per_block.y - 1) / threads_per_block.y
    );

    bool aligned = n % BLOCK_SIZE == 0;
    if (aligned) {
        gemm_kernel_aligned<BLOCK_SIZE><<<blocks, threads_per_block>>>(a_d_input, b_d_input, c_d_output, n);
    } else {
        gemm_kernel_generic<BLOCK_SIZE><<<blocks, threads_per_block>>>(a_d_input, b_d_input, c_d_output, n);
    }

    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());

    check_error(cudaMemcpy(
        c.data(),
        c_d_output,
        elements_num * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    check_error(cudaFree(a_d_input));
    check_error(cudaFree(b_d_input));
    check_error(cudaFree(c_d_output));

    return c;
}
