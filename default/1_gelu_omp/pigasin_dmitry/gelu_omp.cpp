#include "gelu_omp.h"

#include <omp.h>

#include <cmath>


inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.79788456080286541f * (x + 0.044715f * x * x * x)));
}

inline float fastTanh(float x) {
    return 1.f - (2.f / (1.f + std::exp(x * 2.f)));
}

inline float fastGelu(float x) {
    return 0.5f * x * (1.0f + fastTanh(0.79788456080286541f * (x + 0.044715f * x * x * x)));
}

std::vector<float> GeluRef(const std::vector<float> &input) {
    const size_t size = input.size();
    std::vector<float> output(size);

    const float *__restrict in_ptr = input.data();
    float *__restrict out_ptr = output.data();

    for (size_t i = 0; i < size; ++i) {
        out_ptr[i] = gelu(in_ptr[i]);
    }

    return output;
}

std::vector<float> GeluOMP(const std::vector<float> &input) {
    const size_t size = input.size();
    std::vector<float> output(size);

    const float *__restrict in_ptr = input.data();
    float *__restrict out_ptr = output.data();

    #pragma omp parallel for simd
    for (size_t i = 0; i < size; ++i) {
        out_ptr[i] = fastGelu(in_ptr[i]);
    }

    return output;
}
