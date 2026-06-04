#include "gelu_omp.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

inline float tanh(float x) {
    auto exp = std::exp(2 * x);
    return (exp - 1.) / (exp + 1.);
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    constexpr float innerCoeff = M_2_SQRTPI * M_SQRT1_2;
    constexpr float cubeCoeff = innerCoeff * 0.044715;

    std::vector<float> output(input.size());
    const float* inPtr = input.data();
    float* outPtr = output.data();

#pragma omp parallel for
    for(size_t i = 0; i < output.size(); ++i) {
        float x = inPtr[i];
        float cube = x * x * x;
        outPtr[i] = 0.5 * x * (1 + tanh(innerCoeff * x + cubeCoeff * cube));
    }

    return output;
}

#if 0
int main() {
    const size_t input_size = 2000000;
    std::vector<float> input(input_size);

    GeluOMP(input);

    std::vector<double> time_list;
    for (int i = 0; i < 10; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        GeluOMP(input);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());

    std::cout << time << std::endl;
}
#endif
