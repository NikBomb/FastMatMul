#include "goto_matmul.h"

#include "naive_matmul.h"
#include <algorithm>

void Loop4(const double* A, const double* B, double* C, int m, int n, int k, int ldA, int ldB, int ldC, const BlockParams& params) {
	const int kc= std::min(k, params.kc);
    const int num_depth_block = k / kc;
	for (int kblock = 0; kblock < num_depth_block; kblock++) {
	    naive_matmul(&A[kblock * kc], &B[kc * kblock * ldB], C,  m, n, kc, ldA, ldB, ldC);
	}	
}

void goto_matmul(const double* A, const double* B, double* C, int m, int n, int k, int ldA, int ldB, int ldC, const BlockParams& params) {
    // Placeholder implementation until cache-blocked kernel is ready.

    const int num_subblocks = n / params.nc;
    for (int jblock = 0 ; jblock < num_subblocks; ++jblock) { // 5th loop around the microkernel
        Loop4(A, &B[params.nc *jblock], &C[params.nc * jblock], m, params.nc, k, ldA, ldB, ldC, params);
    }
}
