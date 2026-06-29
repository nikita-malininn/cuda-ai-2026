import argparse
import random
from time import perf_counter


import pycuda.driver as cuda
import numpy as np

import layernorm_pycuda as ln


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=1000, help="Number of elements in a row")
    return parser.parse_args()


def generate_input_data(
    size: int,
    low: float = -100.0,
    high: float = 100.0,
) -> list[float]:
    return [random.uniform(low, high) for _ in range(size)]


def benchmark(func, *args, repeat: int = 5, warmup: int = 1):
    """
    Measure function execution time in milliseconds.

    """
    for _ in range(warmup):
        result = func(*args)

    times_ms = []

    for _ in range(repeat):
        start = perf_counter()
        result = func(*args)
        end = perf_counter()
        times_ms.append((end - start) * 1000)

    return result, np.asarray(times_ms)


def compare_results(ref: np.ndarray, actual: np.ndarray):
    ref = np.asarray(ref, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)

    abs_diff = np.abs(ref - actual)

    return {
        "mean_abs_diff": abs_diff.mean(),
        "max_abs_diff": abs_diff.max(),
        "allclose": np.allclose(ref, actual, rtol=1e-3, atol=1e-5),
    }


if __name__ == "__main__":
    args = parse_args()

    dev = cuda.Context.get_device()
    print("GPU:", dev.name())
    print("Compute capability:", dev.compute_capability())

    row_size = args.n
    input_data = generate_input_data(row_size)
    gamma = generate_input_data(row_size)
    beta = generate_input_data(row_size)

    ref_res, ref_times = benchmark(
        ln.layernorm,
        input_data,
        gamma,
        beta,
        row_size,
    )
    pycuda_res, pycuda_times = benchmark(
        ln.layernorm_pycuda,
        input_data,
        gamma,
        beta,
        row_size,
    )

    diff_stats = compare_results(ref_res, pycuda_res)

    elements_num = row_size ** 2
    print("Elements num (^2):", elements_num)
    print("NumPy LayerNorm:")
    print(f"\tmean time: {ref_times.mean():.4f} ms")
    print(f"\tmin time:  {ref_times.min():.4f} ms")
    print(f"\tmax time:  {ref_times.max():.4f} ms")

    print("PyCUDA LayerNorm:")
    print(f"\tmean time: {pycuda_times.mean():.4f} ms")
    print(f"\tmin time:  {pycuda_times.min():.4f} ms")
    print(f"\tmax time:  {pycuda_times.max():.4f} ms")

    print("Difference:")
    print(f"\tmean abs diff: {diff_stats['mean_abs_diff']:.8f}")
    print(f"\tmax abs diff:  {diff_stats['max_abs_diff']:.8f}")
    print(f"\tallclose:      {diff_stats['allclose']}")
