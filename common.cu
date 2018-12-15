#include "common.h"

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int block_size(int thread_per_block, int total_thread){
    return (total_thread + thread_per_block - 1) / thread_per_block;
}

__global__ void trans_unif2exp(int n, float *u, float *thetaSq) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) {
        u[i] = (-1.0f/thetaSq[i]) * logf(u[i]);
    }
}

float *read_mat_col_maj(const char *a, int *row, int *col){
    std::ifstream myfile;
    myfile.open(a);

    myfile >> *row >> *col;
    float *mdata = (float *)malloc(sizeof(float) * (*row) * (*col));

    // Note: column major
    for (int i = 0; i < *row; i++) {       // i-th row
        for (int j = 0; j < *col; j++) {   // j-th col
            myfile >> mdata[j * (*row) + i];
        }
    }
    myfile.close();
    return mdata;
}

int print_mat_col_maj(float *a, int row, int col){
    for (int i = 0; i < row; i++) {       // i-th row
        for (int j = 0; j < col; j++) {   // j-th col
            std::cout <<  a[j * row + i] << "\t";
        }
        std::cout << std::endl;
    }
    return 0;
}

float *read_vec(const char *a, int *n){

    std::ifstream myfile;
    myfile.open(a);
    myfile >> *n;
    float *mdata = (float *)malloc(sizeof(float)* (*n));

    // Note: column major
    for (int i = 0; i < *n; i++) {       // i-th row
        myfile >> mdata[i];
    }
    myfile.close();
    return mdata;
}

int print_vec(float *a, int n){

    // Note: column major
    for (int i = 0; i < n; i++) {       // i-th row
        std::cout << a[i] << std::endl;
    }
    return 0;
}

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS         : return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED : return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED    : return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE   : return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH   : return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR   : return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR  : return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "<unknown>";
}

const char *curandGetErrorString(curandStatus_t error)
{
    switch (error)
    {
        case CURAND_STATUS_SUCCESS                  : return "CURAND_STATUS_SUCCESS";
        case CURAND_STATUS_VERSION_MISMATCH         : return "CURAND_STATUS_VERSION_MISMATCH";
        case CURAND_STATUS_NOT_INITIALIZED          : return "CURAND_STATUS_NOT_INITIALIZED";
        case CURAND_STATUS_ALLOCATION_FAILED        : return "CURAND_STATUS_ALLOCATION_FAILED";
        case CURAND_STATUS_TYPE_ERROR               : return "CURAND_STATUS_TYPE_ERROR";
        case CURAND_STATUS_OUT_OF_RANGE             : return "CURAND_STATUS_OUT_OF_RANGE";
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE      : return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
        case CURAND_STATUS_LAUNCH_FAILURE           : return "CURAND_STATUS_LAUNCH_FAILURE";
        case CURAND_STATUS_PREEXISTING_FAILURE      : return "CURAND_STATUS_PREEXISTING_FAILURE";
        case CURAND_STATUS_INITIALIZATION_FAILED    : return "CURAND_STATUS_INITIALIZATION_FAILED";
        case CURAND_STATUS_ARCH_MISMATCH            : return "CURAND_STATUS_ARCH_MISMATCH";
        case CURAND_STATUS_INTERNAL_ERROR           : return "CURAND_STATUS_INTERNAL_ERROR";
    }
    return "<unknown>";
}

const char *cusolverGetErrorString(cusolverStatus_t error)
{
    switch (error)
    {
        case CUSOLVER_STATUS_SUCCESS                  : return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED          : return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED             : return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE            : return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH            : return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_EXECUTION_FAILED         : return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR           : return "CUSOLVER_STATUS_INTERNAL_ERROR";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    }
    return "<unknown>";
}

int d_print_mat(const float *a,int n,int p){
    float *tmp = (float *)malloc(n*p*sizeof(float));
    CUDA_CALL(cudaMemcpy(tmp,a,n*p*sizeof(float),cudaMemcpyDeviceToHost));
    print_mat_col_maj(tmp,n,p);

    free(tmp);
    return 0;
}

__global__ void shrink_vector(float *d_vec, int n, float * d_scale){
    // TODO: Maybe put d_scale to shared mem?
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n){
        d_vec[tid] /= (*d_scale);
    }
}
