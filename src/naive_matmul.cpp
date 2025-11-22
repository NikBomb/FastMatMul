#include "naive_matmul.h"

void naive_matmul(const double* A, const double* B, double* C, int m, int n, int k,
                  int earlyM, int earlyN, int earlyK) {
    for (int i = 0; i < earlyM; ++i) {
        for (int j = 0; j < earlyN; ++j) {
            for (int p = 0; p < earlyK; ++p) {
                C[n*i + j] += A[k*i + p] * B[n*p + j];            
            }
        }
    }
}
