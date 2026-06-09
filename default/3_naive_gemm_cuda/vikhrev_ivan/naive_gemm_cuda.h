#pragma once

#include <vector>
#include <cmath>

std::vector<float> NaiveGemm(const std::vector<float>& a,
                             const std::vector<float>& b,
                             int n);

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n);

std::vector<float> NaiveGemmCUDASimple(const std::vector<float>& a,
                                    const std::vector<float>& b,
                                    int n);
