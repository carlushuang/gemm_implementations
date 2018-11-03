#include "gemm_opt.h"
#include <stdlib.h>
#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3
#include <immintrin.h>  // FMA

typedef union{
    __m128 v;
    float  f[4];
}v4f_t;

#define BLOCK_K 128
#define BLOCK_M 256

static void InnerKernel( int M, int N, int K,  float alpha,
                                        float *A, int lda, 
                                        float *B, int ldb,
                                        float beta,
                                        float *C, int ldc ){
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

            v4f_t a_val_0, a_val_1, a_val_2, a_val_3, b_val;
            for(k=0;k<K;k++) {

                a_val_0.v  = _mm_load_ps1(ptr_a_0);
                a_val_1.v  = _mm_load_ps1(ptr_a_1);
                a_val_2.v  = _mm_load_ps1(ptr_a_2);
                a_val_3.v  = _mm_load_ps1(ptr_a_3);

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
}

#define MIN(a, b) ( ((a)<(b))?(a):(b) )

static int get_int_value(int def_value, const char * env_var_name){
    if(!env_var_name)
        return def_value;
    char * env = getenv(env_var_name);
    if(env){
        return atoi(env);
    }else
        return def_value;
}

void cblas_sgemm_v7(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB,
                OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
                OPENBLAS_CONST float alpha,
                OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda,
                OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb,
                OPENBLAS_CONST float beta,
                float *C, OPENBLAS_CONST blasint ldc)
{
    int block_k, block_m;
    block_k = get_int_value(BLOCK_K, "BLOCK_K");
    block_m = get_int_value(BLOCK_M, "BLOCK_M");
    // https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_8
    if(Order == CblasRowMajor){
        if(TransA == CblasNoTrans || TransA == CblasConjNoTrans){
            if(TransB == CblasNoTrans|| CblasNoTrans== CblasConjNoTrans){
                if(beta != 1.0f  && beta != 0){
                    int i, j;
                    for(j=0;j<M;j++){
                        for(i=0;i<N;i++){
                            int idx = j*ldc+i;
                            C[idx] *= beta;
                        }
                    }
                }
                int im, ik;
                for(im = 0; im < M; im+=block_m){
                    for(ik = 0; ik < K; ik += block_k){
                        InnerKernel( MIN(M-im, block_m), N, MIN(K-ik, block_k), alpha,
                                        (float*)&A[im*lda + ik], lda, 
                                        (float*)&B[ik*ldb], ldb,
                                        1.0f,                       // force beta tobe 1
                                        &C[im*ldc], ldc); 
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


static void InnerKernel_pack( int M, int N, int K,  float alpha,
                                        float *A, int lda, 
                                        float *B, int ldb,
                                        float beta,
                                        float *C, int ldc ){
    float A_pack[4*K] __attribute__ ((aligned (16)));
    //__builtin_memcpy(A_pack, A, M*K*sizeof(float));
    int m, n, k;
    for(m=0;m<M;m+=4){
        for(n=0;n<N;n+=4){
            v4f_t c_val_r0, c_val_r1, c_val_r2, c_val_r3;
            if( n == 0 ){
                // pack A
                __builtin_memcpy(A_pack     , A+(m+0)*lda, sizeof(float)*K);
                __builtin_memcpy(A_pack+K   , A+(m+1)*lda, sizeof(float)*K);
                __builtin_memcpy(A_pack+K*2 , A+(m+2)*lda, sizeof(float)*K);
                __builtin_memcpy(A_pack+K*3 , A+(m+3)*lda, sizeof(float)*K);
            }

            c_val_r0.v = _mm_setzero_ps();
            c_val_r1.v = _mm_setzero_ps();
            c_val_r2.v = _mm_setzero_ps();
            c_val_r3.v = _mm_setzero_ps();

            float * ptr_a_0 = (float*)&A_pack[0*K];
            float * ptr_a_1 = (float*)&A_pack[1*K];
            float * ptr_a_2 = (float*)&A_pack[2*K];
            float * ptr_a_3 = (float*)&A_pack[3*K];

            float * ptr_b_0 = (float*)&B[n];

            v4f_t a_val_0, a_val_1, a_val_2, a_val_3, b_val;
            for(k=0;k<K;k++) {

                a_val_0.v  = _mm_load_ps1(ptr_a_0);
                a_val_1.v  = _mm_load_ps1(ptr_a_1);
                a_val_2.v  = _mm_load_ps1(ptr_a_2);
                a_val_3.v  = _mm_load_ps1(ptr_a_3);

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
}

void cblas_sgemm_v8(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB,
                OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
                OPENBLAS_CONST float alpha,
                OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda,
                OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb,
                OPENBLAS_CONST float beta,
                float *C, OPENBLAS_CONST blasint ldc)
{
    int block_k, block_m;
    block_k = get_int_value(BLOCK_K, "BLOCK_K");
    block_m = get_int_value(BLOCK_M, "BLOCK_M");
    // https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_4x4_8
    if(Order == CblasRowMajor){
        if(TransA == CblasNoTrans || TransA == CblasConjNoTrans){
            if(TransB == CblasNoTrans|| CblasNoTrans== CblasConjNoTrans){
                if(beta != 1.0f  && beta != 0){
                    int i, j;
                    for(j=0;j<M;j++){
                        for(i=0;i<N;i++){
                            int idx = j*ldc+i;
                            C[idx] *= beta;
                        }
                    }
                }
                int im, ik;
                for(im = 0; im < M; im+=block_m){
                    for(ik = 0; ik < K; ik += block_k){
                        InnerKernel_pack( MIN(M-im, block_m), N, MIN(K-ik, block_k), alpha,
                                        (float*)&A[im*lda + ik], lda, 
                                        (float*)&B[ik*ldb], ldb,
                                        1.0f,                       // force beta tobe 1
                                        &C[im*ldc], ldc); 
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