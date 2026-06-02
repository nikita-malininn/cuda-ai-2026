#pragma once

#include <vector>
#include <cmath>

std::vector<float> Gelu(const std::vector<float>& input);

std::vector<float> GeluCUDANaive(const std::vector<float>& input);

std::vector<float> GeluCUDA(const std::vector<float>& input);
