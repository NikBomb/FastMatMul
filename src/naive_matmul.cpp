#include "naive_matmul.h"

void naive_matmul(const double* A, const double* B, double* C, int m, int n, int k,
                  int ldA, int ldB, int ldC) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int p = 0; p < k; ++p) {
                C[ldC *i + j] += A[ldA*i + p] * B[ldB*p + j];            
            }
        }
    }
}
