#include "common.h"



extern "C"
__global__ void cuda_set(float *u, int n, float value) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) {
        u[i] = value;
    }
}

int main(int argc, char **argv){
    int n = 10;
    if(argc > 1){
        n = atoi(argv[1]);
        printf("%d\n",n);
    }

    int data_bytes = n * sizeof(float);

    // CPU
    float *a = (float *)malloc(data_bytes);
    srand(time(NULL));

    // init
    // printf("%f\n",(float)rand() / RAND_MAX);
    for(int i = 0; i < n; i++){
        a[i] = (float)rand() / RAND_MAX;
    }

    // GPU
    float *d_thetasq;
    float *d_a;
    cudaMalloc((float**)&d_thetasq, data_bytes);
    cudaMalloc((float**)&d_a, data_bytes);
    cudaMemcpy(d_a,a,data_bytes,cudaMemcpyHostToDevice);

    // launch kernel
    dim3 block (1024);
    dim3 grid ((n+block.x-1)/block.x);
    cuda_set<<<grid,block>>>(d_thetasq,n,2.0);

    trans_unif2exp <<<grid,block>>>(n,d_a,d_thetasq);


    // Result back to CPU
    cudaMemcpy(a,d_a,data_bytes,cudaMemcpyDeviceToHost);

    // Print to disk
    std::ofstream out;
    out.open("result_exp.txt");
    for(int i = 0; i < n; i++){
        out << a[i] << "\n";
    }
    out.close();
    system("Rscript test_exponential.R");

    // Free CPU
    free(a);

    // Free GPU
    cudaFree(d_a);
    cudaFree(d_thetasq);
    cudaDeviceReset();
    // Free GPU
    return 0;
}
