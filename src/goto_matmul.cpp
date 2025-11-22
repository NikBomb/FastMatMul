#include "goto_matmul.h"

#include "naive_matmul.h"

void goto_matmul(const double* A, const double* B, double* C, int m, int n, int k, int ldA, int ldB, int ldC, const BlockParams& params) {
    // Placeholder implementation until cache-blocked kernel is ready.
    
    const int num_subblocks = n / params.nc;
    for (int jblock = 0 ; jblock < num_subblocks; ++jblock) {
        naive_matmul(A, &B[params.nc *jblock], &C[params.nc * jblock], m, params.nc, k, ldA, ldB, ldC);
    }
}
