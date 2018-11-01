#!/bin/sh
OPENBLAS_DIR=/opt/OpenBLAS/
CC=gcc
SRC=gemm.c
CFLAGS=" -Wall -I${OPENBLAS_DIR}/include/"
LDFLAGS=" -L${OPENBLAS_DIR}/lib -lopenblas -Wl,-rpath,${OPENBLAS_DIR}/lib"
TARGET=gemm

rm -rf $TARGET
$CC $CFLAGS $SRC $LDFLAGS -o $TARGET
