#include "gelu_omp.h"

#include <cstddef>
#include <cmath>

constexpr float SQRT_2_PI = 0.7978845608028654f;
constexpr float COEFF = 0.044715f;
constexpr float HALF = 0.5f;
constexpr float TWO = 2.0f;
constexpr float ONE = 1.0f;

constexpr float C1 = SQRT_2_PI * COEFF;
constexpr float C2 = SQRT_2_PI;

inline float TanhImpl(float x) noexcept
{
    if (x > 9.0f)
        return ONE;
    if (x < -9.0f)
        return -ONE;

    float exp2x = std::exp(x * TWO);
    return (exp2x - ONE) / (exp2x + ONE);
}

std::vector<float> GeluOMP(const std::vector<float> &input)
{
    const size_t size = input.size();
    const float *__restrict inputPtr = input.data();

    std::vector<float> result(input.size());
    float *__restrict resultPtr = result.data();
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < size; ++i)
    {
        const float x = inputPtr[i];
        const float x2 = x * x;
        const float x3 = x2 * x;
        const float inner = C1 * x3 + C2 * x;

        resultPtr[i] = HALF * x * (ONE + TanhImpl(inner));
    }
    return result;
}