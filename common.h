#ifndef _COMMON_H
#define _COMMON_H

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <assert.h>     /* assert */

#include <time.h>
#include <assert.h>

#include <thrust/version.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/system/cuda/execution_policy.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <cublas.h>
#include <cublas_v2.h>
#include "cusolverDn.h"

double cpuSecond() ;

template <typename T>
struct sqfirst_innerprod
{
  __host__ __device__
  T operator()(const T& x, const T& y) const
  { 
    return x * x * y;
  }
};

template <typename T>
struct x_plus_aysq
{
  const float _a;
  x_plus_aysq(float a = 0):_a(a){};

  __host__ __device__
  T operator()(const T& x, const T& y) const
  { 
    return x + _a * y * y;
  }
};

__global__ void trans_unif2exp(int n, float *u, float *thetaSq);

int block_size(int thread_per_block, int total_thread);

float *read_mat_col_maj(const char *a, int *row, int *col);

int print_mat_col_maj(float *a, int row, int col);

float *read_vec(const char *a, int *n);

int print_vec(float *a, int n);

const char *cublasGetErrorString(cublasStatus_t status);

const char *curandGetErrorString(curandStatus_t error);

const char *cusolverGetErrorString(cusolverStatus_t error);

__global__ void shrink_vector(float *d_vec, int n, float * d_scale);

__global__ void initialize_state(curandState *states);
__global__ void rgamma(curandState *states, float *d_res, float *d_alpha, float *d_beta);

#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    printf("CUDA error:  %s\n", cudaGetErrorString(cudaGetLastError())); \
    exit(EXIT_FAILURE);}} 

#define CURAND_CALL(x) {\
    curandStatus_t error = (x);   \
    if((error) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    printf("CURAND error:  %s\n", curandGetErrorString(error)); \
    exit(EXIT_FAILURE);}}

#define CUBLAS_CALL(x) {\
    cublasStatus_t error = (x); \
    if((error) != CUBLAS_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    printf("CUBLAS error:  %s\n", cublasGetErrorString(error)); \
    exit(EXIT_FAILURE);}}

#define CUSOLVER_CALL(x) {          \
    cusolverStatus_t error = (x); \
    if((error) != CUSOLVER_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    printf("CUSOLVER error:  %s\n", cusolverGetErrorString(error)); \
    exit(EXIT_FAILURE);}}

int d_print_mat(const float *a,int n,int p);

#endif // _COMMON_H
