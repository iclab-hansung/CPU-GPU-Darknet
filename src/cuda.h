#ifndef CUDA_H
#define CUDA_H

#include "darknet.h"

#ifdef GPU

void check_error(cudaError_t status);
void check_error_line(cudaError_t status, int line);
cublasHandle_t blas_handle();
cublasHandle_t blas_handle_a(int idx);
int *cuda_make_int_array(int *x, size_t n);
void cuda_random(float *x_gpu, size_t n);
float cuda_compare(float *x_gpu, float *x, size_t n, char *s);
dim3 cuda_gridsize(size_t n);
void cuda_synchronize(int id, int line);
cudaStream_t usedstream(int id);
#ifdef CUDNN
cudnnHandle_t cudnn_handle();
#endif

#endif
#endif
