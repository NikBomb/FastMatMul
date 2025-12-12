#pragma once

#include <cstddef>

struct BlockParams {
    int mc = 256;
    int kc = 256;
    int nc = 16;
    int mr = 4;
    int nr = 4;
};

void goto_matmul(const double* A, const double* B, double* C, int m, int n, int k, int ldA, int ldB, int ldC, const BlockParams& params);
