import numpy as np
import time
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

layernorm_kernel = """
constexpr int WARP_SIZE = 32;

struct Pair {
    float sum;
    float sq_sum;
};

__device__ __forceinline__ Pair warpReduce(Pair pair) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        pair.sum += __shfl_down_sync(0xffffffff, pair.sum, offset);
        pair.sq_sum += __shfl_down_sync(0xffffffff, pair.sq_sum, offset);
    }
    return pair;
}

__device__ __forceinline__ Pair blockReduce(Pair pair) {
    int t_id = threadIdx.x;
    int warp_id = t_id / WARP_SIZE;
    int lane_id = t_id % WARP_SIZE;

    __shared__ Pair shared_mem[WARP_SIZE];
    pair = warpReduce(pair);
    if (lane_id == 0) {
        shared_mem[warp_id] = pair;
    }
    __syncthreads();

    Pair block_pair;
    block_pair.sum = (t_id < ((blockDim.x + WARP_SIZE - 1) / WARP_SIZE)) ? shared_mem[lane_id].sum : 0.0f;
    block_pair.sq_sum = (t_id < ((blockDim.x + WARP_SIZE - 1) / WARP_SIZE)) ? shared_mem[lane_id].sq_sum : 0.0f;
    if (warp_id == 0) {
        auto pair_block = warpReduce(block_pair);
        if (t_id == 0) {
            shared_mem[0] = pair_block;
        }
    }

   __syncthreads();

    return shared_mem[0];
}

__global__ void layernorm(const float* input, const float* gamma, const float* beta, float* output, int row_size, float eps) {
    int row_ix = blockIdx.x;
    int t_id = threadIdx.x;
    const float* row_input_data = input + row_ix * row_size;
    float* row_output_data = output + row_ix * row_size;

    Pair pair;
    pair.sum = 0.0f;
    pair.sq_sum = 0.0f;
    for (int col = t_id; col < row_size; col += blockDim.x) {
        pair.sum += row_input_data[col];
        pair.sq_sum += row_input_data[col] * row_input_data[col];
    }
    auto [total_sum, total_sq_sum] = blockReduce(pair);

    __shared__ float mean;
    __shared__ float r_variance;
    
    if (t_id == 0) {
        mean = total_sum / row_size;
        float variance = total_sq_sum / row_size - mean * mean;
        r_variance = rsqrtf(variance + eps);
    }
    __syncthreads();
   
    for (int col = t_id; col < row_size; col += blockDim.x) {
        row_output_data[col] = ((row_input_data[col] - mean) * r_variance) * gamma[col] + beta[col];
    }
}
"""

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.

    Parameters
    ----------
    input : list or numpy.ndarray of float
        Flattened matrix in row‑major order. Its length must be divisible by row_size.
    gamma : list or numpy.ndarray of float
        Scale parameter, length = row_size.
    beta : list or numpy.ndarray of float
        Shift parameter, length = row_size.
    row_size : int
        Number of features per row (i.e., number of columns).
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    numpy.ndarray
        Flattened matrix of the same shape as input, containing the row‑wise
        normalized results.
    """
    mod = SourceModule(layernorm_kernel)
    layer_norm = mod.get_function("layernorm")

    input_np = np.asarray(input, dtype=np.float32)
    original_shape = input_np.shape
    input_np = input_np.ravel()
    gamma_np = np.asarray(gamma, dtype=np.float32).ravel()
    beta_np = np.asarray(beta, dtype=np.float32).ravel()

    d_input = cuda.mem_alloc(input_np.nbytes)
    d_gamma = cuda.mem_alloc(gamma_np.nbytes)
    d_beta = cuda.mem_alloc(beta_np.nbytes)
    d_output = cuda.mem_alloc(input_np.nbytes)

    stream = cuda.Stream()

    cuda.memcpy_htod_async(d_input, input_np, stream)
    cuda.memcpy_htod_async(d_gamma, gamma_np, stream)
    cuda.memcpy_htod_async(d_beta, beta_np, stream)

    BLOCK_SIZE = 256
    row_count = input_np.size // row_size
    layer_norm(d_input, d_gamma, d_beta, d_output, np.int32(row_size), np.float32(eps), block=(BLOCK_SIZE, 1, 1), grid=(row_count, 1), stream=stream)
    output = np.empty_like(input_np)
    cuda.memcpy_dtoh_async(output, d_output, stream)

    stream.synchronize()

    d_input.free()
    d_gamma.free()
    d_beta.free()
    d_output.free()

    return output.reshape(original_shape)