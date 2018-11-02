
#include "gemm_opt.h"



#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3
#include <immintrin.h>  // FMA

typedef union{
    __m128 v;
    float  f[4];
}v4f_t;

void cblas_sgemm_v6(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB,
                OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
                OPENBLAS_CONST float alpha,
                OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda,
                OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb,
                OPENBLAS_CONST float beta,
                float *C, OPENBLAS_CONST blasint ldc){
// https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_8
    if(Order == CblasRowMajor){
        if(TransA == CblasNoTrans || TransA == CblasConjNoTrans){
            if(TransB == CblasNoTrans|| CblasNoTrans== CblasConjNoTrans){
                int m, n, k;
                for(m=0;m<M;m+=4){
                    for(n=0;n<N;n+=4){
                        v4f_t c_val_r0, c_val_r1, c_val_r2, c_val_r3;
                        
                        c_val_r0.v = _mm_setzero_ps();
                        c_val_r1.v = _mm_setzero_ps();
                        c_val_r2.v = _mm_setzero_ps();
                        c_val_r3.v = _mm_setzero_ps();

                        float * ptr_a_0 = (float*)&A[(m+0)*lda];
                        float * ptr_a_1 = (float*)&A[(m+1)*lda];
                        float * ptr_a_2 = (float*)&A[(m+2)*lda];
                        float * ptr_a_3 = (float*)&A[(m+3)*lda];

                        float * ptr_b_0 = (float*)&B[n];
                        //float * ptr_b_1 = (float*)&B[n+1];
                        //float * ptr_b_2 = (float*)&B[n+2];
                        //float * ptr_b_3 = (float*)&B[n+3];

                        v4f_t a_val_0, a_val_1, a_val_2, a_val_3, b_val;
                        for(k=0;k<K;k++) {
                            //register float b_val_0, b_val_1, b_val_2, b_val_3;
                            //b_val_0 = B[k*ldb+n];
                            //b_val_1 = B[k*ldb+n+1];
                            //b_val_2 = B[k*ldb+n+2];
                            //b_val_3 = B[k*ldb+n+3];
                            //a_val_0.v  = _mm_set_ps(*ptr_a_0, *ptr_a_0, *ptr_a_0, *ptr_a_0);
                            //a_val_1.v  = _mm_set_ps(*ptr_a_1, *ptr_a_1, *ptr_a_1, *ptr_a_1);
                            //a_val_2.v  = _mm_set_ps(*ptr_a_2, *ptr_a_2, *ptr_a_2, *ptr_a_2);
                            //a_val_3.v  = _mm_set_ps(*ptr_a_3, *ptr_a_3, *ptr_a_3, *ptr_a_3);
                            a_val_0.v  = _mm_load_ps1(ptr_a_0);
                            a_val_1.v  = _mm_load_ps1(ptr_a_1);
                            a_val_2.v  = _mm_load_ps1(ptr_a_2);
                            a_val_3.v  = _mm_load_ps1(ptr_a_3);
                            //b_val.v    = _mm_set_ps(*(ptr_b_0+3), *(ptr_b_0+2), *(ptr_b_0+1), *ptr_b_0);

                            // TODO: ptr must be 16 byte alignment
                            b_val.v    = _mm_load_ps(ptr_b_0);

                            c_val_r0.v = _mm_fmadd_ps(a_val_0.v, b_val.v, c_val_r0.v);

                            //a_val.v    = _mm_set_ps(*ptr_a_1, *ptr_a_1, *ptr_a_1, *ptr_a_1);
                            c_val_r1.v = _mm_fmadd_ps(a_val_1.v, b_val.v, c_val_r1.v);

                            //a_val.v    = _mm_set_ps(*ptr_a_2, *ptr_a_2, *ptr_a_2, *ptr_a_2);
                            c_val_r2.v = _mm_fmadd_ps(a_val_2.v, b_val.v, c_val_r2.v);

                            //a_val.v    = _mm_set_ps(*ptr_a_3, *ptr_a_3, *ptr_a_3, *ptr_a_3);
                            c_val_r3.v = _mm_fmadd_ps(a_val_3.v, b_val.v, c_val_r3.v);

                            ptr_a_0++;
                            ptr_a_1++;
                            ptr_a_2++;
                            ptr_a_3++;

                            ptr_b_0 += ldb;
                            //ptr_b_1 += ldb;
                            //ptr_b_2 += ldb;
                            //ptr_b_3 += ldb;
                        } 
                        C[(m+0)*ldc+n+0] = c_val_r0.f[0]*alpha + C[(m+0)*ldc+n+0]*beta;
                        C[(m+1)*ldc+n+0] = c_val_r1.f[0]*alpha + C[(m+1)*ldc+n+0]*beta;
                        C[(m+2)*ldc+n+0] = c_val_r2.f[0]*alpha + C[(m+2)*ldc+n+0]*beta;
                        C[(m+3)*ldc+n+0] = c_val_r3.f[0]*alpha + C[(m+3)*ldc+n+0]*beta;

                        C[(m+0)*ldc+n+1] = c_val_r0.f[1]*alpha + C[(m+0)*ldc+n+1]*beta;
                        C[(m+1)*ldc+n+1] = c_val_r1.f[1]*alpha + C[(m+1)*ldc+n+1]*beta;
                        C[(m+2)*ldc+n+1] = c_val_r2.f[1]*alpha + C[(m+2)*ldc+n+1]*beta;
                        C[(m+3)*ldc+n+1] = c_val_r3.f[1]*alpha + C[(m+3)*ldc+n+1]*beta;

                        C[(m+0)*ldc+n+2] = c_val_r0.f[2]*alpha + C[(m+0)*ldc+n+2]*beta;
                        C[(m+1)*ldc+n+2] = c_val_r1.f[2]*alpha + C[(m+1)*ldc+n+2]*beta;
                        C[(m+2)*ldc+n+2] = c_val_r2.f[2]*alpha + C[(m+2)*ldc+n+2]*beta;
                        C[(m+3)*ldc+n+2] = c_val_r3.f[2]*alpha + C[(m+3)*ldc+n+2]*beta;

                        C[(m+0)*ldc+n+3] = c_val_r0.f[3]*alpha + C[(m+0)*ldc+n+3]*beta;
                        C[(m+1)*ldc+n+3] = c_val_r1.f[3]*alpha + C[(m+1)*ldc+n+3]*beta;
                        C[(m+2)*ldc+n+3] = c_val_r2.f[3]*alpha + C[(m+2)*ldc+n+3]*beta;
                        C[(m+3)*ldc+n+3] = c_val_r3.f[3]*alpha + C[(m+3)*ldc+n+3]*beta;
                    }
                }
            }else{

            }
        }else{

        }
    } else {
        //
    }
}