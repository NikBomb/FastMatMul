#include "goto_matmul.h"

#include "naive_matmul.h"
#include <algorithm>
#include <immintrin.h>


void kernel_col(const double* A, const double* B, double* C, int m, int n, int k, int ldA, int ldB, int ldC){

   /* Declare vector registers to hold 4x4 C and load them */
  __m256d gamma_0123_0 = _mm256_setr_pd(C[0*ldC + 0], C[1*ldC + 0],C[2*ldC + 0], C[3*ldC + 0]);
  __m256d gamma_0123_1 = _mm256_setr_pd(C[0*ldC + 1], C[1*ldC + 1],C[2*ldC + 1], C[3*ldC + 1]);
  __m256d gamma_0123_2 = _mm256_setr_pd(C[0*ldC + 2], C[1*ldC + 2],C[2*ldC + 2], C[3*ldC + 2]);
  __m256d gamma_0123_3 = _mm256_setr_pd(C[0*ldC + 3], C[1*ldC + 3],C[2*ldC + 3], C[3*ldC + 3]);

    for ( int p=0; p<k; p++ ){
    /* Declare vector register for load/broadcasting beta( p,j ) */
    __m256d beta_p_j;
    
    /* Declare a vector register to hold the current column of A and load
       it with the four elements of that column. */
    __m256d alpha_0123_p = _mm256_setr_pd(A[0*ldA + p], A[1*ldA + p],A[2*ldA + p], A[3*ldA + p]);;

    /* Load/broadcast beta( p,0 ). */
    beta_p_j = _mm256_broadcast_sd( &B[p*ldB] );
    
    /* update the first column of C with the current column of A times
       beta ( p,0 ) */
    gamma_0123_0 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_0 );
    
    /* REPEAT for second, third, and fourth columns of C.  Notice that the 
       current column of A needs not be reloaded. */

    /* Load/broadcast beta( p,1 ). */
    beta_p_j = _mm256_broadcast_sd( &B[p*ldB + 1] );
    
    /* update the second column of C with the current column of A times
       beta ( p,1 ) */
    gamma_0123_1 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_1 );

    /* Load/broadcast beta( p,2 ). */
    beta_p_j = _mm256_broadcast_sd( &B[p*ldB + 2] );
    
    /* update the third column of C with the current column of A times
       beta ( p,2 ) */
    gamma_0123_2 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_2 );

    /* Load/broadcast beta( p,3 ). */
    beta_p_j = _mm256_broadcast_sd( &B[p*ldB + 3] );
    
    /* update the fourth column of C with the current column of A times
       beta ( p,3 ) */
    gamma_0123_3 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_3 );
  }
  
  /* Store the updated results */
  double tmp[4];
  _mm256_storeu_pd(tmp, gamma_0123_0);
  C[0*ldC + 0] = tmp[0];
  C[1*ldC + 0] = tmp[1];
  C[2*ldC + 0] = tmp[2];
  C[3*ldC + 0] = tmp[3];
  _mm256_storeu_pd(tmp, gamma_0123_1);
  C[0*ldC + 1] = tmp[0];
  C[1*ldC + 1] = tmp[1];
  C[2*ldC + 1] = tmp[2];
  C[3*ldC + 1] = tmp[3];
  _mm256_storeu_pd(tmp, gamma_0123_2);
  C[0*ldC + 2] = tmp[0];
  C[1*ldC + 2] = tmp[1];
  C[2*ldC + 2] = tmp[2];
  C[3*ldC + 2] = tmp[3];
  _mm256_storeu_pd(tmp, gamma_0123_3);
  C[0*ldC + 3] = tmp[0];
  C[1*ldC + 3] = tmp[1];
  C[2*ldC + 3] = tmp[2];
  C[3*ldC + 3] = tmp[3];


}

// Computes C[0..3][0..3] += A[0..3][0..k-1] * B[0..k-1][0..3], row-major.
inline void kernel_row(const double* A, const double* B, double* C, int m, int n, int k, int ldA, int ldB, int ldC) {
    // Load four rows of C contiguously
    __m256d c0 = _mm256_loadu_pd(C + 0*ldC);
    __m256d c1 = _mm256_loadu_pd(C + 1*ldC);
    __m256d c2 = _mm256_loadu_pd(C + 2*ldC);
    __m256d c3 = _mm256_loadu_pd(C + 3*ldC);

    for (int p = 0; p < k; ++p) {
        // Load p-th row of B (4 contiguous cols)
        __m256d b = _mm256_loadu_pd(B + p*ldB);

        // Broadcast p-th element of each A row
        __m256d a0 = _mm256_broadcast_sd(A + 0*ldA + p);
        __m256d a1 = _mm256_broadcast_sd(A + 1*ldA + p);
        __m256d a2 = _mm256_broadcast_sd(A + 2*ldA + p);
        __m256d a3 = _mm256_broadcast_sd(A + 3*ldA + p);

        // FMA into each C row
        c0 = _mm256_fmadd_pd(a0, b, c0);
        c1 = _mm256_fmadd_pd(a1, b, c1);
        c2 = _mm256_fmadd_pd(a2, b, c2);
        c3 = _mm256_fmadd_pd(a3, b, c3);
    }

    // Store C rows back
    _mm256_storeu_pd(C + 0*ldC, c0);
    _mm256_storeu_pd(C + 1*ldC, c1);
    _mm256_storeu_pd(C + 2*ldC, c2);
    _mm256_storeu_pd(C + 3*ldC, c3);
}


void Loop1(const double* A, const double* B, double* C, int m, int n, int k, int ldA, int ldB, int ldC, const BlockParams& params) {
    const int mr = std::min(m, params.mr);
    const int num_row_block = m / mr;
    for (int iblock = 0; iblock < num_row_block; iblock++) {
        kernel_row(&A[mr * iblock * ldA], B, &C[mr * iblock * ldC], mr, n, k, ldA, ldB, ldC);
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

void Loop4(const double* A, const double* B, double* C, int m, int n, int k, int ldA, int ldB, int ldC, const BlockParams& params) {
	const int kc= std::min(k, params.kc);
    const int num_depth_block = k / kc;
	for (int kblock = 0; kblock < num_depth_block; kblock++) {
	    Loop3(&A[kblock * kc], &B[kc * kblock * ldB], C,  m, n, kc, ldA, ldB, ldC, params);
	}	
}


void goto_matmul(const double* A, const double* B, double* C, int m, int n, int k, int ldA, int ldB, int ldC, const BlockParams& params) {
    // Placeholder implementation until cache-blocked kernel is ready.
    const int num_subblocks = n / params.nc;
    for (int jblock = 0 ; jblock < num_subblocks; ++jblock) { // 5th loop around the microkernel
        Loop4(A, &B[params.nc *jblock], &C[params.nc * jblock], m, params.nc, k, ldA, ldB, ldC, params);
    }
}
