#include "goto_matmul.h"

#include "naive_matmul.h"
#include <algorithm>
#include <immintrin.h>
#include <vector>


// 4x8 row-major micro-kernel: C[4x8] += A[4xk] * B[kx8]
inline void kernel_row_4x8(const double* A, const double* B, double* C, int k, int ldA, int ldB, int ldC) {
    __m256d c0_0 = _mm256_loadu_pd(C + 0*ldC + 0);
    __m256d c0_1 = _mm256_loadu_pd(C + 0*ldC + 4);
    __m256d c1_0 = _mm256_loadu_pd(C + 1*ldC + 0);
    __m256d c1_1 = _mm256_loadu_pd(C + 1*ldC + 4);
    __m256d c2_0 = _mm256_loadu_pd(C + 2*ldC + 0);
    __m256d c2_1 = _mm256_loadu_pd(C + 2*ldC + 4);
    __m256d c3_0 = _mm256_loadu_pd(C + 3*ldC + 0);
    __m256d c3_1 = _mm256_loadu_pd(C + 3*ldC + 4);

    for (int p = 0; p < k; ++p) {
        __m256d b0 = _mm256_loadu_pd(B + p*ldB + 0); // cols 0..3
        __m256d b1 = _mm256_loadu_pd(B + p*ldB + 4); // cols 4..7

        __m256d a0 = _mm256_broadcast_sd(A + 0*ldA + p);
        c0_0 = _mm256_fmadd_pd(a0, b0, c0_0);
        c0_1 = _mm256_fmadd_pd(a0, b1, c0_1);

        __m256d a1 = _mm256_broadcast_sd(A + 1*ldA + p);
        c1_0 = _mm256_fmadd_pd(a1, b0, c1_0);
        c1_1 = _mm256_fmadd_pd(a1, b1, c1_1);

        __m256d a2 = _mm256_broadcast_sd(A + 2*ldA + p);
        c2_0 = _mm256_fmadd_pd(a2, b0, c2_0);
        c2_1 = _mm256_fmadd_pd(a2, b1, c2_1);

        __m256d a3 = _mm256_broadcast_sd(A + 3*ldA + p);
        c3_0 = _mm256_fmadd_pd(a3, b0, c3_0);
        c3_1 = _mm256_fmadd_pd(a3, b1, c3_1);
    }

    _mm256_storeu_pd(C + 0*ldC + 0, c0_0);
    _mm256_storeu_pd(C + 0*ldC + 4, c0_1);
    _mm256_storeu_pd(C + 1*ldC + 0, c1_0);
    _mm256_storeu_pd(C + 1*ldC + 4, c1_1);
    _mm256_storeu_pd(C + 2*ldC + 0, c2_0);
    _mm256_storeu_pd(C + 2*ldC + 4, c2_1);
    _mm256_storeu_pd(C + 3*ldC + 0, c3_0);
    _mm256_storeu_pd(C + 3*ldC + 4, c3_1);
}


void Loop1(const double* A, const double* B, double* C, int m, int n, int k, int ldA, int ldB, int ldC, const BlockParams& params) {
    const int mr = std::min(m, params.mr);
    const int num_row_block = m / mr;
    for (int iblock = 0; iblock < num_row_block; iblock++) {
        const double* A_block = &A[mr * iblock * ldA];
        double* C_block = &C[mr * iblock * ldC];
        if (n == params.nr && mr == params.mr) {
            kernel_row_4x8(A_block, B, C_block, k, ldA, ldB, ldC);
        } else {
            naive_matmul(A_block, B, C_block, mr, n, k, ldA, ldB, ldC);
        }
    }
}


void Loop2(const double* A, const double* B, double* C, int m, int n, int k, int ldA, int ldB, int ldC, const BlockParams& params) {
    const int nr = std::min(n, params.nr);
    const int num_col_block = n / nr;
    for (int jblock = 0; jblock < num_col_block; jblock++) {
        Loop1(A, &B[nr * jblock], &C[nr * jblock], m, nr, k, ldA, ldB, ldC, params);
    }
}

void Loop3(const double* A, const double* B, double* C, int m, int n, int k, int ldA, int ldB, int ldC, const BlockParams& params) {
    const int mc = std::min(m, params.mc);
    const int num_row_block = m / mc;
    for (int iblock = 0; iblock < num_row_block; iblock++) {
        //naive_matmul(&A[mc * iblock * ldA], B, &C[mc * iblock * ldC], mc, n, k, ldA, ldB, ldC);
        Loop2(&A[mc * iblock * ldA], B, &C[mc * iblock * ldC], mc, n, k, ldA, ldB, ldC, params);
    }
}

void pack_B_panel(const double* B, int ldB, int kc, int nc, std::vector<double>& packed) {
    packed.resize(static_cast<std::size_t>(kc) * static_cast<std::size_t>(nc));
    for (int p = 0; p < kc; ++p) {
        const double* src = B + p * ldB;
        double* dst = packed.data() + static_cast<std::size_t>(p) * static_cast<std::size_t>(nc);
        std::copy(src, src + nc, dst);
    }
}

void Loop4(const double* A, const double* B, double* C, int m, int n, int k, int ldA, int ldB, int ldC, const BlockParams& params) {
	const int kc= std::min(k, params.kc);
    const int num_depth_block = k / kc;
    std::vector<double> packed_B;
	for (int kblock = 0; kblock < num_depth_block; kblock++) {
        const double* B_panel = &B[kc * kblock * ldB];
        pack_B_panel(B_panel, ldB, kc, n, packed_B);
        const int packed_ldB = n;
	    Loop3(&A[kblock * kc], packed_B.data(), C,  m, n, kc, ldA, packed_ldB, ldC, params);
	}	
}


void goto_matmul(const double* A, const double* B, double* C, int m, int n, int k, int ldA, int ldB, int ldC, const BlockParams& params) {
    const int nc = std::min(params.nc, n);
    for (int jblock = 0; jblock < n; jblock += nc) { // 5th loop around the microkernel
        const int cur_n = std::min(nc, n - jblock);
        Loop4(A, &B[jblock], &C[jblock], m, cur_n, k, ldA, ldB, ldC, params);
    }
}
