#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "softmax_cuda.h"

std::vector<float> SoftmaxRef(const std::vector<float>& input,
                              int row_count) {
    const int row_size =
        static_cast<int>(input.size()) / row_count;

    std::vector<float> output(input.size());

    for (int row = 0; row < row_count; ++row) {
        const int base = row * row_size;

        float row_max = -std::numeric_limits<float>::infinity();

        for (int i = 0; i < row_size; ++i) {
            row_max = std::max(row_max, input[base + i]);
        }

        float sum = 0.0f;

        for (int i = 0; i < row_size; ++i) {
            float v = std::exp(input[base + i] - row_max);
            output[base + i] = v;
            sum += v;
        }

        for (int i = 0; i < row_size; ++i) {
            output[base + i] /= sum;
        }
    }

    return output;
}

template <typename Func>
double Benchmark(Func&& f, int iterations = 5) {
    double best = 1e100;

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        auto result = f();

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        best = std::min(best, elapsed.count());

        volatile float sink = result[result.size() / 2];
        (void)sink;
    }

    return best;
}

int main() {
    constexpr int ROW_COUNT = 4096;
    constexpr int ROW_SIZE  = 1024;

    std::vector<float> input(ROW_COUNT * ROW_SIZE);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.f, 5.f);

    for (auto& x : input) {
        x = dist(rng);
    }

    std::cout << "Matrix size: "
              << ROW_COUNT
              << " x "
              << ROW_SIZE
              << "\n\n";

    auto ref_time = Benchmark([&]() {
        return SoftmaxRef(input, ROW_COUNT);
    });

    auto cuda_time = Benchmark([&]() {
        return SoftmaxCUDA(input, ROW_COUNT);
    });

    auto ref  = SoftmaxRef(input, ROW_COUNT);
    auto cuda = SoftmaxCUDA(input, ROW_COUNT);

    float max_error = 0.0f;

    for (size_t i = 0; i < ref.size(); ++i) {
        max_error =
            std::max(max_error,
                     std::abs(ref[i] - cuda[i]));
    }

    float max_row_sum_error = 0.0f;

    for (int row = 0; row < ROW_COUNT; ++row) {
        float sum = 0.0f;

        for (int j = 0; j < ROW_SIZE; ++j) {
            sum += cuda[row * ROW_SIZE + j];
        }

        max_row_sum_error =
            std::max(max_row_sum_error,
                     std::abs(sum - 1.0f));
    }

    std::cout << "Reference         : " << ref_time << " s\n";
    std::cout << "CUDA              : " << cuda_time << " s\n";
    std::cout << "Speedup           : " << ref_time / cuda_time << "x\n";
    std::cout << "Max abs error     : " << max_error << '\n';
    std::cout << "Max row sum error : " << max_row_sum_error << '\n';

    return 0;
}