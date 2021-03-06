#!/bin/sh
OPENBLAS_DIR=/opt/OpenBLAS/
CC=gcc
SRC="gemm.c gemm_opt.c gemm_opt_2.c"
CFLAGS=" -Wall -O2 -I${OPENBLAS_DIR}/include/ -mfma -msse -msse2"
LDFLAGS=" -L${OPENBLAS_DIR}/lib -lopenblas -lm -Wl,-rpath,${OPENBLAS_DIR}/lib"
TARGET=gemm

rm -rf $TARGET
$CC $CFLAGS $SRC $LDFLAGS -o $TARGET
