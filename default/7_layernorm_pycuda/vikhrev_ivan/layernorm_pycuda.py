import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np


def layernorm(input, gamma, beta, row_size, eps=1e-5):
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
    mat = np.array(input).reshape(-1, row_size)
    mean = np.mean(mat, axis=1, keepdims=True)
    var = np.var(mat, axis=1, keepdims=True)

    normalized = (mat - mean)/ np.sqrt(var + eps)
    y = gamma * normalized + beta
    return y.reshape(-1)


BLOCK_SIZE = 256


layernorm_kernel_code = f"""
#define BLOCK_SIZE {BLOCK_SIZE}
__global__ void layernorm_kernel(float* input, float* gamma, float* beta, float eps, float* output, int rows_num, int cols_num) {{
    int row_id = blockIdx.x;
    int thread_id = threadIdx.x;

    if (row_id >= rows_num) {{
        return;
    }}

    __shared__ float shared_sum[BLOCK_SIZE];
    __shared__ float row_mean;
    __shared__ float row_var;

    // -------------------------
    // 1. Mean
    // -------------------------
    float local_sum = 0.f;
    for (int j = thread_id; j < cols_num; j += blockDim.x) {{
        local_sum += input[row_id * cols_num + j];
    }}

    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {{
        if (threadIdx.x < stride) {{
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }}

        __syncthreads();
    }}

    if (thread_id == 0) {{
        row_mean = shared_sum[0] / cols_num;
    }}
    __syncthreads();

    // -------------------------
    // 2. Variance
    // -------------------------
    float local_var_sum = 0.0f;

    for (int j = thread_id; j < cols_num; j += blockDim.x) {{
        float diff = input[row_id * cols_num + j] - row_mean;
        local_var_sum += diff * diff;
    }}

    shared_sum[thread_id] = local_var_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {{
        if (thread_id < stride) {{
            shared_sum[thread_id] += shared_sum[thread_id + stride];
        }}
        __syncthreads();
    }}

    if (thread_id == 0) {{
        row_var = shared_sum[0] / cols_num;
    }}
    __syncthreads();

    // -------------------------
    // 3. Normalize + scale + shift
    // -------------------------
    float inv_std = rsqrtf(row_var + eps);
    for (int j = threadIdx.x; j < cols_num; j += blockDim.x) {{
        int idx = row_id * cols_num + j;
        output[idx] = gamma[j] * (input[idx] - row_mean) * inv_std + beta[j];
    }}
    __syncthreads();
}}
"""


_module = SourceModule(layernorm_kernel_code)
_layernorm_kernel = _module.get_function("layernorm_kernel")


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

    input = np.array(input, dtype=np.float32)
    gamma = np.array(gamma, dtype=np.float32)
    beta = np.array(beta, dtype=np.float32)
    output = np.empty_like(input)

    if input.size % row_size != 0:
        raise ValueError("input size must be divisible by row_size")

    if gamma.size != row_size:
        raise ValueError("gamma size must be equal to row_size")

    if beta.size != row_size:
        raise ValueError("beta size must be equal to row_size")

    rows = input.size // row_size
    cols = row_size

    d_input = cuda.mem_alloc(input.nbytes)
    d_gamma = cuda.mem_alloc(gamma.nbytes)
    d_beta = cuda.mem_alloc(beta.nbytes)
    d_output = cuda.mem_alloc(output.nbytes)

    cuda.memcpy_htod(d_input, input)
    cuda.memcpy_htod(d_gamma, gamma)
    cuda.memcpy_htod(d_beta, beta)

    _layernorm_kernel(d_input, d_gamma, d_beta, np.float32(eps), d_output, np.int32(rows), np.int32(cols), block=(BLOCK_SIZE, 1, 1), grid=(row_size, 1, 1))
    cuda.Context.synchronize()

    cuda.memcpy_dtoh(output, d_output)
    d_input.free()
    d_gamma.free()
    d_beta.free()
    d_output.free()

    return output
