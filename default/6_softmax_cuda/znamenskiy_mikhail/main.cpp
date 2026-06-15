#include <cstddef>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>

#include <softmax_cuda.h>

namespace {
    std::vector<float> SoftmaxRef(const std::vector<float>& input, int row_count) {
        std::vector<float> output(input.size());
        const int col_count = input.size() / row_count;
        std::vector<float> expInRow(col_count);
        for (int row = 0; row < row_count; ++row) {
            float maxInRow = *std::max_element(input.begin(), input.end());
            float sumInRow = 0.0f;

            for (int col = 0; col < col_count; ++col) {
                expInRow[col] = std::exp(input[row * col_count + col] - maxInRow);
                sumInRow += expInRow[col];
            }
            for (int col = 0; col < col_count; ++col) {
                output[row * col_count + col] = expInRow[col] / sumInRow;
            }
        }

        return output;
    }
}

int main() {

constexpr size_t ROWS = 1024;
constexpr size_t COLS = 1024;
constexpr size_t SIZE = ROWS * COLS;

std::vector<float> input(SIZE);

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dis(-10.f, 10.f);
std::cout << "Generating random data" << std::endl;
for (size_t n = 0; n < SIZE; ++n) {
    input[n] = dis(gen);
    }
std::cout << "Generating DONE" << std::endl;

std::cout << "Ref calculations" << std::endl;
std::chrono::steady_clock::time_point beginRef = std::chrono::steady_clock::now();
auto outputRef = SoftmaxRef(input, ROWS);
std::chrono::steady_clock::time_point endRef = std::chrono::steady_clock::now();
std::cout << "Ref calculations DONE" << std::endl;
std::cout << "Time REF = " << std::chrono::duration_cast<std::chrono::microseconds>(endRef - beginRef).count() << "[us]" << std::endl;

std::cout << "Warming up" << std::endl;
SoftmaxCUDA(input, ROWS);
std::cout << "Warming up DONE" << std::endl;

std::cout << "Measurements" << std::endl;
std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
auto output = SoftmaxCUDA(input, ROWS);
std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
std::cout << "Measurements done" << std::endl;
std::cout << "Time OPT = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;

std::cout << "Accuracy check" << std::endl;
float error = 0.0f;
for (size_t n = 0; n < SIZE; ++n) {
    error = std::max(std::abs(output[n] - outputRef[n]), error);
    if (std::isnan(error)) {
        std::cout << "NAN error - index = " << n << " result = " << output[n] << " ref = " << outputRef[n] << std::endl;
        return 1;
    }
}
std::cout << "Accuracy check FINISHED" << std::endl;
std::cout << "Max error = " << error << std::endl;

return 0;
}
