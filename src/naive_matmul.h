#pragma once

#include <cstddef>

void naive_matmul(const double* A, const double* B, double* C, int m, int n, int k, int ldA, int ldB, int ldC);

