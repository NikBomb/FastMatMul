#include "goto_matmul.h"

#include "naive_matmul.h"

void goto_matmul(const double* A, const double* B, double* C, int n, const BlockParams& params) {
    (void)params;
    // Placeholder implementation until cache-blocked kernel is ready.
    naive_matmul(A, B, C, n);
}
