#include <stdio.h>
#include <cblas.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>

#include "gemm_opt.h"

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


/**********************************************************************/

#define MAT_M  512
#define MAT_N  256
#define MAT_K  512

#define MAT_ALPHA .5
#define MAT_BETA  1.2

/*
 * C = alpha*A*B + beta*C
 *
 * A, m by k (width:k, height:m)
 * B, k by n (width:n, height:k)
 * C, m by n (width:n, height:m)
 *
 */

#define LAYOUT_ROW_MAJOR 1
#define LAYOUT_COL_MAJOR 2

#define TRANS_NO_TRANS      100
#define TRANS_TRANS         101
#define TRANS_CONJ_TRANS    102
#define TRANS_CONJ_NO_TRANS 103

static const char * get_layout_str(int layout){
    if(layout == LAYOUT_ROW_MAJOR)
        return "CblasRowMajor";
    if(layout == LAYOUT_COL_MAJOR)
        return "CblasColMajor";
    return "n/a major";
}
static const char * get_trans_str(int trans){
    if(trans == TRANS_NO_TRANS)
        return "CblasNoTrans";
    if(trans == TRANS_TRANS)
        return "CblasTrans";
    if(trans == TRANS_CONJ_TRANS)
        return "CblasConjTrans";
    if(trans == TRANS_CONJ_NO_TRANS)
        return "CblasConjNoTrans";
    return "n/a trans";
}

typedef struct {
    float * data;
    int     row;
    int     col;
    int     layout;
    int     trans;
} matrix_f32_t;

// value is 0-1
static void rand_matrix_f32(float * v, int elem) {
    int i;

    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }

    for(i=0;i<elem;i++){
        v[i] = ((float)(rand() % 100)) / 100.0f;
    }
}

int matrix_f32_leading(matrix_f32_t * mat, int is_c /* if is output buffer*/){
    if(is_c){
        if(mat->layout == LAYOUT_ROW_MAJOR)
            return mat->col;
        else
            return mat->row;
    }
    if(mat->layout == LAYOUT_ROW_MAJOR){
        if(mat->trans == TRANS_NO_TRANS)
            return mat->col;
        return mat->row;
    }else{
        if(mat->trans == TRANS_NO_TRANS)
            return mat->row;
        return mat->col;
    }
}

#define MEM_ALIGN_BYTE 16

matrix_f32_t * matrix_f32_create(int row, int col, int layout, int trans){
    matrix_f32_t * mat = (matrix_f32_t*)malloc(sizeof(matrix_f32_t));
    mat->row = row;
    mat->col = col;
    mat->layout = layout;
    mat->trans = trans;
    mat->data = (float *) aligned_alloc(MEM_ALIGN_BYTE, sizeof(float) * row * col);
    rand_matrix_f32(mat->data, row * col);
    return mat;
}
matrix_f32_t * matrix_f32_create_copy(matrix_f32_t * mat_rhs){
    matrix_f32_t * mat = (matrix_f32_t*)malloc(sizeof(matrix_f32_t));
    mat->row = mat_rhs->row;
    mat->col = mat_rhs->col;
    mat->layout = mat_rhs->layout;
    mat->trans = mat_rhs->trans;
    mat->data = (float *) aligned_alloc(MEM_ALIGN_BYTE, sizeof(float) * mat->row * mat->col);
    memcpy(mat->data, mat_rhs->data, sizeof(float) * mat->row * mat->col);
    return mat;
}

void matric_f32_dump_raw(matrix_f32_t * mat){
    int i;
    if(!mat || !mat->data)
        return;

    for(i=0; i<mat->col*mat->row; i++)
        printf("%f ", mat->data[i]);
    printf("\n");
}

void matric_f32_dump(matrix_f32_t * mat){
    int c, r, col, row;
    int index;
    if(!mat || !mat->data)
        return;
    printf("matrix row:%d, col:%d, layout:%s, trans:%s\n", mat->row, mat->col, get_layout_str(mat->layout), get_trans_str(mat->trans));
    if(mat->trans == TRANS_NO_TRANS || mat->trans == TRANS_CONJ_TRANS){
         if(mat->layout == LAYOUT_ROW_MAJOR){
            col = mat->col;
            row = mat->row;
            for(r=0;r<row;r++){
                for(c=0;c<col;c++){
                    index = r*col+c;
                    printf("%f ", mat->data[index]);
                }
                printf("\n");
            }
         }else{
            col = mat->col;
            row = mat->row;
            for(r=0;r<row;r++){
                for(c=0;c<col;c++){
                    index = r+c*row;
                    printf("%f ", mat->data[index]);
                }
                printf("\n");
            }
         }
    }else{
        // TODO
        if(mat->layout == LAYOUT_ROW_MAJOR){
            col = mat->col;
            row = mat->row;
            for(r=0;r<row;r++){
                for(c=0;c<col;c++){
                    index = r+c*row;
                    printf("%f ", mat->data[index]);
                }
                printf("\n");
            }
         }else{
            col = mat->col;
            row = mat->row;
            for(r=0;r<row;r++){
                for(c=0;c<col;c++){
                    index = r*col+c;
                    printf("%f ", mat->data[index]);
                }
                printf("\n");
            }
         }
    }
}

void matrix_f32_free(matrix_f32_t * mat){
    if(mat){
        if(mat->data)
            free(mat->data);
        free(mat);
    }
}

CBLAS_ORDER matrix_layout_blas(int layout){
    if(layout == LAYOUT_ROW_MAJOR)
        return CblasRowMajor;
    //if(layout == LAYOUT_COL_MAJOR)
    //TODO: validation
    return CblasColMajor;
}
CBLAS_TRANSPOSE matrix_trans_blas(int trans){
    switch(trans){
        case  TRANS_NO_TRANS:
            return CblasNoTrans;
        case  TRANS_TRANS:
            return CblasTrans;
        case TRANS_CONJ_TRANS:
            return CblasConjTrans;
        case TRANS_CONJ_NO_TRANS:
            return CblasConjNoTrans;
        default:
            return CblasConjNoTrans;
    }
}


static void dump_mat_raw(float * mat, int elem){
    int i;
    for(i=0;i<elem;i++)
        printf("%f ", mat[i]);
    printf("\n");
}



// c stupid implementation
void cblas_sgemm_v0(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB,
                OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
                OPENBLAS_CONST float alpha,
                OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda,
                OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb,
                OPENBLAS_CONST float beta,
                float *C, OPENBLAS_CONST blasint ldc){

    if(Order == CblasRowMajor){
        int m, n, k;
        for(m=0;m<M;m++){
            for(n=0;n<N;n++){
                int c_idx = m*ldc+n;
                float c_val = C[c_idx] * beta;
                for(k=0;k<K;k++){
                    int a_idx = (TransA == CblasNoTrans || TransA == CblasConjNoTrans)?
                                (m*lda+k) : (k*lda+m);
                    int b_idx = (TransB == CblasNoTrans || TransB == CblasConjNoTrans)?
                                (k*ldb+n) : (n*ldb+k);
                    c_val += alpha * A[a_idx] * B[b_idx];
                }
                C[c_idx] = c_val;
            }
        }
    } else {
        int m, n, k;
        for( n=0;n<N;n++ ){
            for( m=0;m<M;m++ ){
                int c_idx = n*ldc+m;
                float c_val = C[c_idx] * beta;
                for( k=0;k<K;k++ ) {
                    int a_idx = (TransA == CblasNoTrans || TransA == CblasConjNoTrans)?
                                (k*lda+m) : (m*lda+k);
                    int b_idx = (TransB == CblasNoTrans || TransB == CblasConjNoTrans)?
                                (n*ldb+k) : (k*ldb+n);
                    c_val += alpha * A[a_idx] * B[b_idx];
                }
                C[c_idx] = c_val;
            }
        }
    }
}

void cblas_sgemm_v1(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB,
                OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
                OPENBLAS_CONST float alpha,
                OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda,
                OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb,
                OPENBLAS_CONST float beta,
                float *C, OPENBLAS_CONST blasint ldc){
// https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_1x4_5
    if(Order == CblasRowMajor){
        if(TransA == CblasNoTrans || TransA == CblasConjNoTrans){
            if(TransB == CblasNoTrans|| CblasNoTrans== CblasConjNoTrans){
                int m, n, k;
                for(m=0;m<M;m+=4){
                    for(n=0;n<N;n++){
                        // TODO: ensure m is divided by 4 !!!
                        register float c_val_0 = 0;
                        register float c_val_1 = 0;
                        register float c_val_2 = 0;
                        register float c_val_3 = 0;
                        for(k=0;k<K;k++){
                            c_val_0 += A[(m+0)*lda+k]*B[k*ldb+n];
                            c_val_1 += A[(m+1)*lda+k]*B[k*ldb+n];
                            c_val_2 += A[(m+2)*lda+k]*B[k*ldb+n];
                            c_val_3 += A[(m+3)*lda+k]*B[k*ldb+n];
                        }
                        C[m*ldc+n]     = c_val_0*alpha + C[m*ldc+n]*beta;
                        C[(m+1)*ldc+n] = c_val_1*alpha + C[(m+1)*ldc+n]*beta;
                        C[(m+2)*ldc+n] = c_val_2*alpha + C[(m+2)*ldc+n]*beta;
                        C[(m+3)*ldc+n] = c_val_3*alpha + C[(m+3)*ldc+n]*beta;
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

void cblas_sgemm_v2(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB,
                OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
                OPENBLAS_CONST float alpha,
                OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda,
                OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb,
                OPENBLAS_CONST float beta,
                float *C, OPENBLAS_CONST blasint ldc){
// https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_1x4_7
    if(Order == CblasRowMajor){
        if(TransA == CblasNoTrans || TransA == CblasConjNoTrans){
            if(TransB == CblasNoTrans|| CblasNoTrans== CblasConjNoTrans){
                int m, n, k;
                for(m=0;m<M;m+=4){
                    for(n=0;n<N;n++){
                        // TODO: ensure m is divided by 4 !!!
                        register float c_val_0 = 0;
                        register float c_val_1 = 0;
                        register float c_val_2 = 0;
                        register float c_val_3 = 0;
                        float * ptr_a_0 = (float*)&A[(m+0)*lda];
                        float * ptr_a_1 = (float*)&A[(m+1)*lda];
                        float * ptr_a_2 = (float*)&A[(m+2)*lda];
                        float * ptr_a_3 = (float*)&A[(m+3)*lda];
                        for(k=0;k<K;k++){
                            register float b_val = B[k*ldb+n];
                            c_val_0 += *ptr_a_0++ * b_val;
                            c_val_1 += *ptr_a_1++ * b_val;
                            c_val_2 += *ptr_a_2++ * b_val;
                            c_val_3 += *ptr_a_3++ * b_val;
                        }
                        C[m*ldc+n]     = c_val_0*alpha + C[m*ldc+n]*beta;
                        C[(m+1)*ldc+n] = c_val_1*alpha + C[(m+1)*ldc+n]*beta;
                        C[(m+2)*ldc+n] = c_val_2*alpha + C[(m+2)*ldc+n]*beta;
                        C[(m+3)*ldc+n] = c_val_3*alpha + C[(m+3)*ldc+n]*beta;
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

void cblas_sgemm_v3(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB,
                OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
                OPENBLAS_CONST float alpha,
                OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda,
                OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb,
                OPENBLAS_CONST float beta,
                float *C, OPENBLAS_CONST blasint ldc){
// https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_1x4_9
    if(Order == CblasRowMajor){
        if(TransA == CblasNoTrans || TransA == CblasConjNoTrans){
            if(TransB == CblasNoTrans|| CblasNoTrans== CblasConjNoTrans){
                int m, n, k;
                for(m=0;m<M;m+=4){
                    for(n=0;n<N;n++){
                        // TODO: ensure m is divided by 4 !!!
                        register float c_val_0 = 0;
                        register float c_val_1 = 0;
                        register float c_val_2 = 0;
                        register float c_val_3 = 0;
                        float * ptr_a_0 = (float*)&A[(m+0)*lda];
                        float * ptr_a_1 = (float*)&A[(m+1)*lda];
                        float * ptr_a_2 = (float*)&A[(m+2)*lda];
                        float * ptr_a_3 = (float*)&A[(m+3)*lda];
                        for(k=0;k<K;k+=4){
                            register float b_val;
                            b_val = B[k*ldb+n];
                            c_val_0 += *ptr_a_0 * b_val;
                            c_val_1 += *ptr_a_1 * b_val;
                            c_val_2 += *ptr_a_2 * b_val;
                            c_val_3 += *ptr_a_3 * b_val;

                            b_val = B[(k+1)*ldb+n];
                            c_val_0 += *(ptr_a_0+1) * b_val;
                            c_val_1 += *(ptr_a_1+1) * b_val;
                            c_val_2 += *(ptr_a_2+1) * b_val;
                            c_val_3 += *(ptr_a_3+1) * b_val;

                            b_val = B[(k+2)*ldb+n];
                            c_val_0 += *(ptr_a_0+2) * b_val;
                            c_val_1 += *(ptr_a_1+2) * b_val;
                            c_val_2 += *(ptr_a_2+2) * b_val;
                            c_val_3 += *(ptr_a_3+2) * b_val;

                            b_val = B[(k+3)*ldb+n];
                            c_val_0 += *(ptr_a_0+3) * b_val;
                            c_val_1 += *(ptr_a_1+3) * b_val;
                            c_val_2 += *(ptr_a_2+3) * b_val;
                            c_val_3 += *(ptr_a_3+3) * b_val;

                            // pointer use indirect addressing
                            ptr_a_0 += 4;
                            ptr_a_1 += 4;
                            ptr_a_2 += 4;
                            ptr_a_3 += 4;
                        }
                        C[m*ldc+n]     = c_val_0*alpha + C[m*ldc+n]*beta;
                        C[(m+1)*ldc+n] = c_val_1*alpha + C[(m+1)*ldc+n]*beta;
                        C[(m+2)*ldc+n] = c_val_2*alpha + C[(m+2)*ldc+n]*beta;
                        C[(m+3)*ldc+n] = c_val_3*alpha + C[(m+3)*ldc+n]*beta;
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

void cblas_sgemm_v4(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB,
                OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
                OPENBLAS_CONST float alpha,
                OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda,
                OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb,
                OPENBLAS_CONST float beta,
                float *C, OPENBLAS_CONST blasint ldc){
// https://github.com/flame/how-to-optimize-gemm/wiki/Optimization_1x4_9
    if(Order == CblasRowMajor){
        if(TransA == CblasNoTrans || TransA == CblasConjNoTrans){
            if(TransB == CblasNoTrans|| CblasNoTrans== CblasConjNoTrans){
                int m, n, k;
                for(m=0;m<M;m+=4){
                    for(n=0;n<N;n+=4){
                        // TODO: ensure m is divided by 4 !!!
                        register float
                        c_val_0_0, c_val_0_1, c_val_0_2, c_val_0_3,
                        c_val_1_0, c_val_1_1, c_val_1_2, c_val_1_3,
                        c_val_2_0, c_val_2_1, c_val_2_2, c_val_2_3,
                        c_val_3_0, c_val_3_1, c_val_3_2, c_val_3_3;

                        c_val_0_0=0; c_val_0_1=0; c_val_0_2=0; c_val_0_3=0;
                        c_val_1_0=0; c_val_1_1=0; c_val_1_2=0; c_val_1_3=0;
                        c_val_2_0=0; c_val_2_1=0; c_val_2_2=0; c_val_2_3=0;
                        c_val_3_0=0; c_val_3_1=0; c_val_3_2=0; c_val_3_3=0;

                        float * ptr_a_0 = (float*)&A[(m+0)*lda];
                        float * ptr_a_1 = (float*)&A[(m+1)*lda];
                        float * ptr_a_2 = (float*)&A[(m+2)*lda];
                        float * ptr_a_3 = (float*)&A[(m+3)*lda];

                        float * ptr_b_0 = (float*)&B[n];
                        float * ptr_b_1 = (float*)&B[n+1];
                        float * ptr_b_2 = (float*)&B[n+2];
                        float * ptr_b_3 = (float*)&B[n+3];
                        for(k=0;k<K;k++) {
                            //register float b_val_0, b_val_1, b_val_2, b_val_3;
                            //b_val_0 = B[k*ldb+n];
                            //b_val_1 = B[k*ldb+n+1];
                            //b_val_2 = B[k*ldb+n+2];
                            //b_val_3 = B[k*ldb+n+3];

                            c_val_0_0 += *ptr_a_0 * *ptr_b_0;
                            c_val_1_0 += *ptr_a_1 * *ptr_b_0;
                            c_val_2_0 += *ptr_a_2 * *ptr_b_0;
                            c_val_3_0 += *ptr_a_3 * *ptr_b_0;

                            c_val_0_1 += *ptr_a_0 * *ptr_b_1;
                            c_val_1_1 += *ptr_a_1 * *ptr_b_1;
                            c_val_2_1 += *ptr_a_2 * *ptr_b_1;
                            c_val_3_1 += *ptr_a_3 * *ptr_b_1;

                            c_val_0_2 += *ptr_a_0 * *ptr_b_2;
                            c_val_1_2 += *ptr_a_1 * *ptr_b_2;
                            c_val_2_2 += *ptr_a_2 * *ptr_b_2;
                            c_val_3_2 += *ptr_a_3 * *ptr_b_2;

                            c_val_0_3 += *ptr_a_0 * *ptr_b_3;
                            c_val_1_3 += *ptr_a_1 * *ptr_b_3;
                            c_val_2_3 += *ptr_a_2 * *ptr_b_3;
                            c_val_3_3 += *ptr_a_3 * *ptr_b_3;

                            ptr_a_0++;
                            ptr_a_1++;
                            ptr_a_2++;
                            ptr_a_3++;

                            ptr_b_0 += ldb;
                            ptr_b_1 += ldb;
                            ptr_b_2 += ldb;
                            ptr_b_3 += ldb;
                        }
                        C[(m+0)*ldc+n+0] = c_val_0_0*alpha + C[(m+0)*ldc+n+0]*beta;
                        C[(m+1)*ldc+n+0] = c_val_1_0*alpha + C[(m+1)*ldc+n+0]*beta;
                        C[(m+2)*ldc+n+0] = c_val_2_0*alpha + C[(m+2)*ldc+n+0]*beta;
                        C[(m+3)*ldc+n+0] = c_val_3_0*alpha + C[(m+3)*ldc+n+0]*beta;

                        C[(m+0)*ldc+n+1] = c_val_0_1*alpha + C[(m+0)*ldc+n+1]*beta;
                        C[(m+1)*ldc+n+1] = c_val_1_1*alpha + C[(m+1)*ldc+n+1]*beta;
                        C[(m+2)*ldc+n+1] = c_val_2_1*alpha + C[(m+2)*ldc+n+1]*beta;
                        C[(m+3)*ldc+n+1] = c_val_3_1*alpha + C[(m+3)*ldc+n+1]*beta;

                        C[(m+0)*ldc+n+2] = c_val_0_2*alpha + C[(m+0)*ldc+n+2]*beta;
                        C[(m+1)*ldc+n+2] = c_val_1_2*alpha + C[(m+1)*ldc+n+2]*beta;
                        C[(m+2)*ldc+n+2] = c_val_2_2*alpha + C[(m+2)*ldc+n+2]*beta;
                        C[(m+3)*ldc+n+2] = c_val_3_2*alpha + C[(m+3)*ldc+n+2]*beta;

                        C[(m+0)*ldc+n+3] = c_val_0_3*alpha + C[(m+0)*ldc+n+3]*beta;
                        C[(m+1)*ldc+n+3] = c_val_1_3*alpha + C[(m+1)*ldc+n+3]*beta;
                        C[(m+2)*ldc+n+3] = c_val_2_3*alpha + C[(m+2)*ldc+n+3]*beta;
                        C[(m+3)*ldc+n+3] = c_val_3_3*alpha + C[(m+3)*ldc+n+3]*beta;
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

void cblas_sgemm_v5(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB,
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
                        // TODO: ensure m is divided by 4 !!!
                        register float
                        c_val_0_0, c_val_0_1, c_val_0_2, c_val_0_3,
                        c_val_1_0, c_val_1_1, c_val_1_2, c_val_1_3,
                        c_val_2_0, c_val_2_1, c_val_2_2, c_val_2_3,
                        c_val_3_0, c_val_3_1, c_val_3_2, c_val_3_3;

                        register float
                        a_val_0, a_val_1, a_val_2, a_val_3,
                        b_val_0, b_val_1, b_val_2, b_val_3;

                        c_val_0_0=0; c_val_0_1=0; c_val_0_2=0; c_val_0_3=0;
                        c_val_1_0=0; c_val_1_1=0; c_val_1_2=0; c_val_1_3=0;
                        c_val_2_0=0; c_val_2_1=0; c_val_2_2=0; c_val_2_3=0;
                        c_val_3_0=0; c_val_3_1=0; c_val_3_2=0; c_val_3_3=0;

                        float * ptr_a_0 = (float*)&A[(m+0)*lda];
                        float * ptr_a_1 = (float*)&A[(m+1)*lda];
                        float * ptr_a_2 = (float*)&A[(m+2)*lda];
                        float * ptr_a_3 = (float*)&A[(m+3)*lda];

                        float * ptr_b_0 = (float*)&B[n];
                        float * ptr_b_1 = (float*)&B[n+1];
                        float * ptr_b_2 = (float*)&B[n+2];
                        float * ptr_b_3 = (float*)&B[n+3];
                        for(k=0;k<K;k++) {
                            //register float b_val_0, b_val_1, b_val_2, b_val_3;
                            //b_val_0 = B[k*ldb+n];
                            //b_val_1 = B[k*ldb+n+1];
                            //b_val_2 = B[k*ldb+n+2];
                            //b_val_3 = B[k*ldb+n+3];
                            a_val_0 = *ptr_a_0;
                            a_val_1 = *ptr_a_1;
                            a_val_2 = *ptr_a_2;
                            a_val_3 = *ptr_a_3;

                            b_val_0 = *ptr_b_0;
                            b_val_1 = *ptr_b_1;
                            b_val_2 = *ptr_b_2;
                            b_val_3 = *ptr_b_3;

                            c_val_0_0 += a_val_0 * b_val_0;
                            c_val_1_0 += a_val_1 * b_val_0;
                            c_val_2_0 += a_val_2 * b_val_0;
                            c_val_3_0 += a_val_3 * b_val_0;

                            c_val_0_1 += a_val_0 * b_val_1;
                            c_val_1_1 += a_val_1 * b_val_1;
                            c_val_2_1 += a_val_2 * b_val_1;
                            c_val_3_1 += a_val_3 * b_val_1;

                            c_val_0_2 += a_val_0 * b_val_2;
                            c_val_1_2 += a_val_1 * b_val_2;
                            c_val_2_2 += a_val_2 * b_val_2;
                            c_val_3_2 += a_val_3 * b_val_2;

                            c_val_0_3 += a_val_0 * b_val_3;
                            c_val_1_3 += a_val_1 * b_val_3;
                            c_val_2_3 += a_val_2 * b_val_3;
                            c_val_3_3 += a_val_3 * b_val_3;

                            ptr_a_0++;
                            ptr_a_1++;
                            ptr_a_2++;
                            ptr_a_3++;

                            ptr_b_0 += ldb;
                            ptr_b_1 += ldb;
                            ptr_b_2 += ldb;
                            ptr_b_3 += ldb;
                        }
                        C[(m+0)*ldc+n+0] = c_val_0_0*alpha + C[(m+0)*ldc+n+0]*beta;
                        C[(m+1)*ldc+n+0] = c_val_1_0*alpha + C[(m+1)*ldc+n+0]*beta;
                        C[(m+2)*ldc+n+0] = c_val_2_0*alpha + C[(m+2)*ldc+n+0]*beta;
                        C[(m+3)*ldc+n+0] = c_val_3_0*alpha + C[(m+3)*ldc+n+0]*beta;

                        C[(m+0)*ldc+n+1] = c_val_0_1*alpha + C[(m+0)*ldc+n+1]*beta;
                        C[(m+1)*ldc+n+1] = c_val_1_1*alpha + C[(m+1)*ldc+n+1]*beta;
                        C[(m+2)*ldc+n+1] = c_val_2_1*alpha + C[(m+2)*ldc+n+1]*beta;
                        C[(m+3)*ldc+n+1] = c_val_3_1*alpha + C[(m+3)*ldc+n+1]*beta;

                        C[(m+0)*ldc+n+2] = c_val_0_2*alpha + C[(m+0)*ldc+n+2]*beta;
                        C[(m+1)*ldc+n+2] = c_val_1_2*alpha + C[(m+1)*ldc+n+2]*beta;
                        C[(m+2)*ldc+n+2] = c_val_2_2*alpha + C[(m+2)*ldc+n+2]*beta;
                        C[(m+3)*ldc+n+2] = c_val_3_2*alpha + C[(m+3)*ldc+n+2]*beta;

                        C[(m+0)*ldc+n+3] = c_val_0_3*alpha + C[(m+0)*ldc+n+3]*beta;
                        C[(m+1)*ldc+n+3] = c_val_1_3*alpha + C[(m+1)*ldc+n+3]*beta;
                        C[(m+2)*ldc+n+3] = c_val_2_3*alpha + C[(m+2)*ldc+n+3]*beta;
                        C[(m+3)*ldc+n+3] = c_val_3_3*alpha + C[(m+3)*ldc+n+3]*beta;
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

// https://devtalk.nvidia.com/default/topic/482834/how-to-compute-gflops-for-gemm-blas/
static unsigned long long sgemm_flop(unsigned long long M, unsigned long long N, unsigned long long K,
    float alpha, float beta)
{
    if(alpha == 1.f && beta == 0.f){
        // M*N*K mul, M*N*(K-1) add
        return M*N*(2*K-1);
    }
    if(alpha == 1.f && beta != 0.f){
        // M*N*K mul, M*N*(K-1) add, M*N beta mul, M*N beta add
        return M*N*(2*K + 1);
    }
    if(alpha != 1.f && beta == 0.f){
        // M*N*K mul, M*N*(K-1) add, M*N alpha mul
        return M*N*(2*K);
    }

    // alpha != 1.f, beta != 0.f
    // M*N*K mul, M*N*(K-1) add, M*N alpha mul, M*N beta mul, M*N beta add
    return M*N*(2*K+2);
}

#define LOOP 100
#define SKIP_LOOP 10
#define SLEEP_USEC 50*1000

#define BENCH_SGEMM(m, n, k, alpha, beta, mat_a, mat_b, mat_c, loop, sgemm_func ) \
    do{                                 \
        const int total_loop = loop;    \
        int l;                          \
                                        \
        usleep(SLEEP_USEC);             \
        double start_time = 0;                                      \
        for(l=0;l<total_loop;l++){                                  \
            if(l==SKIP_LOOP){                                       \
                start_time = what_time_is_it_now();                 \
            }                                                       \
            sgemm_func(matrix_layout_blas(mat_a->layout),           \
                matrix_trans_blas(mat_a->trans),                    \
                matrix_trans_blas(mat_b->trans),                    \
                m, n, k,                                            \
                alpha,                                              \
                mat_a->data, matrix_f32_leading(mat_a, 0) /*k*/,    \
                mat_b->data, matrix_f32_leading(mat_b, 0) /*n*/,    \
                beta,                                               \
                mat_c->data, matrix_f32_leading(mat_c, 1) /*n*/);   \
        }                                                           \
        double cost_time = what_time_is_it_now()-start_time;        \
        unsigned long long flop = sgemm_flop((unsigned long long)m, \
                                            (unsigned long long)n,  \
                                            (unsigned long long)k,  \
                                            (float)alpha,           \
                                            (float)beta);           \
        double cost_sec = cost_time/(total_loop-SKIP_LOOP);         \
        double gflops = flop/cost_sec/(1e9);                               \
        printf("%16s: gflops: %f, cost_per_loop:%fms\n",             \
                #sgemm_func,   gflops,                               \
                (cost_time/(total_loop-SKIP_LOOP))*1000   );   \
    }while(0)

static int get_int_value(int def_value, const char * env_var_name){
    if(!env_var_name)
        return def_value;
    char * env = getenv(env_var_name);
    if(env){
        return atoi(env);
    }else
        return def_value;
}

int main(int argc, char ** argv){
    matrix_f32_t * mat_a, * mat_b, * mat_c, *mat_c_2;
    int m, n, k;
    float alpha, beta;
    int trans, layout;
    int loop;
    int run_ver;
    int verbose;
    int need_valid;

    trans = TRANS_NO_TRANS;
    layout = LAYOUT_ROW_MAJOR;
    if(argc>2){
        // TODO: valide
        trans = atoi(argv[1]);
        layout = atoi(argv[2]);
    }

    m = get_int_value(MAT_M, "M");
    n = get_int_value(MAT_N, "N");
    k = get_int_value(MAT_K, "K");

    run_ver = get_int_value(-1, "VER");
    verbose = get_int_value(0, "V");
    need_valid = get_int_value(0, "VALID");

    alpha = MAT_ALPHA;
    beta = MAT_BETA;
    loop = get_int_value(LOOP, "LOOP");

    printf("M:%d, N:%d, K:%d, ALPHA:%f, BETA:%f, LOOP:%d, TRANS:%s, LAYOUT:%s, ver:%d, openblas:%s\n",
            m,n,k,alpha, beta, loop, get_trans_str(trans), get_layout_str(layout), run_ver,
                (openblas_get_parallel()==1)?"mul":"sig");
    openblas_set_num_threads(1);// force openblas single thread

    mat_a = matrix_f32_create(m, k, layout, trans);
    mat_b = matrix_f32_create(k, n, layout, trans);
    mat_c = matrix_f32_create(m, n, layout, trans);
    mat_c_2 =  matrix_f32_create_copy(mat_c);

    if(verbose){
        printf("mat_a:\n");
        matric_f32_dump(mat_a);
        printf("mat_b:\n");
        matric_f32_dump(mat_b);
        printf("mat_c:\n");
        matric_f32_dump(mat_c);
        printf("mat_c_2:\n");
        matric_f32_dump(mat_c_2);
    }
    BENCH_SGEMM(m, n, k, alpha, beta, mat_a, mat_b, mat_c, loop, cblas_sgemm);

    //memcpy(mat_c->data, mat_c_2->data, mat_c->col*mat_c->row*sizeof(float));

    if(run_ver == -1){
        memcpy(mat_c->data, mat_c_2->data, mat_c->col*mat_c->row*sizeof(float));
        BENCH_SGEMM(m, n, k, alpha, beta, mat_a, mat_b, mat_c, loop, cblas_sgemm_v1);

        memcpy(mat_c->data, mat_c_2->data, mat_c->col*mat_c->row*sizeof(float));
        BENCH_SGEMM(m, n, k, alpha, beta, mat_a, mat_b, mat_c, loop, cblas_sgemm_v2);

        memcpy(mat_c->data, mat_c_2->data, mat_c->col*mat_c->row*sizeof(float));
        BENCH_SGEMM(m, n, k, alpha, beta, mat_a, mat_b, mat_c, loop, cblas_sgemm_v3);

        memcpy(mat_c->data, mat_c_2->data, mat_c->col*mat_c->row*sizeof(float));
        BENCH_SGEMM(m, n, k, alpha, beta, mat_a, mat_b, mat_c, loop, cblas_sgemm_v4);

        memcpy(mat_c->data, mat_c_2->data, mat_c->col*mat_c->row*sizeof(float));
        BENCH_SGEMM(m, n, k, alpha, beta, mat_a, mat_b, mat_c, loop, cblas_sgemm_v5);

        memcpy(mat_c->data, mat_c_2->data, mat_c->col*mat_c->row*sizeof(float));
        BENCH_SGEMM(m, n, k, alpha, beta, mat_a, mat_b, mat_c, loop, cblas_sgemm_v6);

        memcpy(mat_c->data, mat_c_2->data, mat_c->col*mat_c->row*sizeof(float));
        BENCH_SGEMM(m, n, k, alpha, beta, mat_a, mat_b, mat_c, loop, cblas_sgemm_v7);

        memcpy(mat_c->data, mat_c_2->data, mat_c->col*mat_c->row*sizeof(float));
        BENCH_SGEMM(m, n, k, alpha, beta, mat_a, mat_b, mat_c, loop, cblas_sgemm_v8);

        memcpy(mat_c->data, mat_c_2->data, mat_c->col*mat_c->row*sizeof(float));
        BENCH_SGEMM(m, n, k, alpha, beta, mat_a, mat_b, mat_c, loop, cblas_sgemm_v9);
    }
    // per run process
    else if(run_ver == 0)
        BENCH_SGEMM(m, n, k, alpha, beta, mat_a, mat_b, mat_c_2, loop, cblas_sgemm_v0);
    else if(run_ver == 1)
        BENCH_SGEMM(m, n, k, alpha, beta, mat_a, mat_b, mat_c_2, loop, cblas_sgemm_v1);
    else if(run_ver == 2)
        BENCH_SGEMM(m, n, k, alpha, beta, mat_a, mat_b, mat_c_2, loop, cblas_sgemm_v2);
    else if(run_ver == 3)
        BENCH_SGEMM(m, n, k, alpha, beta, mat_a, mat_b, mat_c_2, loop, cblas_sgemm_v3);
    else if(run_ver == 4)
        BENCH_SGEMM(m, n, k, alpha, beta, mat_a, mat_b, mat_c_2, loop, cblas_sgemm_v4);
    else if(run_ver == 5)
        BENCH_SGEMM(m, n, k, alpha, beta, mat_a, mat_b, mat_c_2, loop, cblas_sgemm_v5);
    else if(run_ver == 6)
        BENCH_SGEMM(m, n, k, alpha, beta, mat_a, mat_b, mat_c_2, loop, cblas_sgemm_v6);
    else if(run_ver == 7)
        BENCH_SGEMM(m, n, k, alpha, beta, mat_a, mat_b, mat_c_2, loop, cblas_sgemm_v7);
    else if(run_ver == 8)
        BENCH_SGEMM(m, n, k, alpha, beta, mat_a, mat_b, mat_c_2, loop, cblas_sgemm_v8);
    else if(run_ver == 9)
        BENCH_SGEMM(m, n, k, alpha, beta, mat_a, mat_b, mat_c_2, loop, cblas_sgemm_v9);

    if(need_valid){
        //validate
        int x;
        int invalid_cnt = 0;
        for(x=0;x<(mat_c->col*mat_c->row);x++){
            float delta = fabsf(mat_c->data[x] - mat_c_2->data[x]);
            //printf("%3d golda:%f -- impl:%f, delta:%f\n", x, mat_c->data[x], mat_c_2->data[x], delta);
            if(delta > 0.0001f){
                if(invalid_cnt<10)
                    printf("## not valid at idx:%-4d, golden:%f, impl:%f, delta:%f\n",
                        x, mat_c->data[x],  mat_c_2->data[x], delta);
                invalid_cnt++;
            }
        }

        if(!invalid_cnt)
            printf("## impl valid!\n");
    }

    if(verbose){
        printf("mat_c:\n");
        matric_f32_dump(mat_c);
        printf("mat_c_2:\n");
        matric_f32_dump(mat_c_2);

        matrix_f32_free(mat_a);
        matrix_f32_free(mat_b);
        matrix_f32_free(mat_c);
        matrix_f32_free(mat_c_2);
    }

    return 0;
}
