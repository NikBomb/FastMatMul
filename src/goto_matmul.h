#pragma once

#include <cstddef>

struct BlockParams {
    int mc = 256;
    int kc = 256;
    int nc = 512;
    int mr = 8;
    int nr = 4;
};

void goto_matmul(const double* A, const double* B, double* C, int n, const BlockParams& params);
