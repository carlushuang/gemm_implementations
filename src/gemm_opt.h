#ifndef __GEMM_OPT_H
#define __GEMM_OPT_H
#include <cblas.h>

void cblas_sgemm_v6(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB,
                OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
                OPENBLAS_CONST float alpha,
                OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda,
                OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb,
                OPENBLAS_CONST float beta,
                float *C, OPENBLAS_CONST blasint ldc);

#endif
