#pragma once

#include <cstddef>

void naive_matmul(const double* A, const double* B, double* C, int m, int n, int k, int earlyM, int earlyN, int earlyK);

