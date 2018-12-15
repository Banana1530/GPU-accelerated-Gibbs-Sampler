#include "common.h"


#define DEBUG


int main(int argc, char **argv)
{
    int N = 1000;
    if(argc > 1){
        N = atoi(argv[1]);
    }

    // CPU data
    int n;        // n
    int p;        // p
    int p_lambdaSqInv;
    int n_z;
    
    float tauSqInv = 1;
    float *X   = read_mat_col_maj("./test_data/data_X.txt",&n,&p);
    printf("Data size: n = %d, p = %d.\n",n,p);

    float *lambdaSqInv = read_vec("./test_data/data_lambdaSqInv.txt",&p_lambdaSqInv); 
    assert(p_lambdaSqInv == p);

    float *z = read_vec("./test_data/data_z.txt",&n_z); 
    assert(n_z == n);

    // Prepare
    float one  = 1.0;
    float zero = 0.0;
    cublasHandle_t cublas_handle = 0;
    cusolverDnHandle_t cusolver_handle = 0;
    curandGenerator_t randGen;
    CUSOLVER_CALL(cusolverDnCreate(&cusolver_handle));
    CUBLAS_CALL(cublasCreate(&cublas_handle));
    CURAND_CALL(curandCreateGenerator(&randGen,
                CURAND_RNG_PSEUDO_PHILOX4_32_10));              // NOTE: Choice of rng could affect results
                                                                // e.g., variance does not match theoretical result.

    // GPU data
    float *d_X;
    float *d_XtX;
    float *d_lambdaSqInv;
    float *d_z; 
    float *d_beta;


    CUDA_CALL(cudaMalloc((void **)&d_X,   n*p*sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&d_XtX, p*p*sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&d_lambdaSqInv, p*sizeof(float))); 
    CUDA_CALL(cudaMalloc((void **)&d_z,    n*sizeof(float))); 
    CUDA_CALL(cudaMalloc((void **)&d_beta, p*sizeof(float))); 

    // copy data
    CUBLAS_CALL(cublasSetMatrix(n, p, sizeof(float),
                X, n, d_X, n));
    CUDA_CALL(cudaMemcpy(d_lambdaSqInv,lambdaSqInv,p*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_z,z,n*sizeof(float),cudaMemcpyHostToDevice));

    // d_XtX = d_X.t * d_X
    CUBLAS_CALL(cublasSgemm(cublas_handle,CUBLAS_OP_T,CUBLAS_OP_N,p,p,n,&one,
                d_X,n,d_X,n,&zero,d_XtX,p));
#ifdef DEBUG
    printf("XtX == \n");
    d_print_mat(d_XtX,p,p);
#endif

    // working space
    float *d_SigmaInv;
    float *d_mu;
    CUDA_CALL(cudaMalloc((void **)&d_SigmaInv,   p*p*sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&d_mu,         p*sizeof(float)));

   
    float *betas = (float *) malloc(N * p * sizeof(float));
    for(int i = 0; i < N; i++){

        /* 
        * Step2: 
        * d_SigmaInv = d_XtX
        * d_SigmaInv += tauSqInv .* d_lambdaSqInv
        * yj <- alpha * xj + yj
        */
        CUDA_CALL(cudaMemcpy(d_SigmaInv,d_XtX,p*p*sizeof(float),cudaMemcpyDeviceToDevice));
        CUBLAS_CALL(cublasSaxpy(cublas_handle, p, &tauSqInv, d_lambdaSqInv, 1, d_SigmaInv, p+1));
#ifdef DEBUG
        printf("=After step 2: d_SigmaInv == \n");
        d_print_mat(d_SigmaInv,p,p);
#endif 


        /* 
        * Step3: Cholesky decomposition
        * Upper(d_SigmaInv) = R, where d_SigmaInv = R R^T
        */
        int cholWorkspaceSize;
        int *devInfo;
        float *d_cholWorkspace;
        CUDA_CALL(cudaMalloc((void **)&devInfo,sizeof(int)));

        CUSOLVER_CALL(cusolverDnSpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_UPPER, 
                      p, d_SigmaInv, 1, &cholWorkspaceSize));
        CUDA_CALL(cudaMalloc((void **)&d_cholWorkspace, cholWorkspaceSize*sizeof(float)));
        CUSOLVER_CALL(cusolverDnSpotrf(cusolver_handle, CUBLAS_FILL_MODE_UPPER,p, d_SigmaInv,
                      p,d_cholWorkspace, cholWorkspaceSize*sizeof(float),devInfo ));
        CUDA_CALL(cudaFree(d_cholWorkspace));
        CUDA_CALL(cudaFree(devInfo));
#ifdef DEBUG
        printf("==After step 3: d_SigmaInv == \n");
        d_print_mat(d_SigmaInv,p,p);
#endif 


        /*
        * Step 4: Generate uniforms
        * beta ~iid N(0,1)
        */
        CURAND_CALL(curandGenerateNormal(randGen, d_beta, p, 0, 1));
#ifdef DEBUG
        printf("==After step 4: d_beta == \n");
        d_print_mat(d_beta,p,1);
#endif


        /* 
        * Step 5: beta = Rs = solve(upper(SigmaInv,beta))
        */
        CUBLAS_CALL(cublasStrsv(cublas_handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 
                    p, d_SigmaInv, p, d_beta, 1));
#ifdef DEBUG
        printf("==After step 5: d_beta == \n");
        d_print_mat(d_beta,p,1);
#endif


        /* 
        * Step 6: mu = Xt z 
        */
        CUBLAS_CALL(cublasSgemv(cublas_handle,CUBLAS_OP_T,n,p,&one,d_X,n,d_z,1,&zero,d_mu,1));
#ifdef DEBUG
        printf("==After step 6 : d_z == \n");
        d_print_mat(d_z,n,1);
        printf("d_mu == \n");
        d_print_mat(d_mu,p,1);
        printf("d_X == \n");
        d_print_mat(d_X,n,p);
#endif
      

        /* 
        * Step 7: mu = Sigma mu = Solve(d_SigmaInv,mu) 
        */
        CUSOLVER_CALL(cusolverDnSpotrs(cusolver_handle, CUBLAS_FILL_MODE_UPPER, 
                      p, 1, d_SigmaInv, p, d_mu, p, devInfo));
#ifdef DEBUG
        printf("=After step 7: d_mu == \n");
        d_print_mat(d_mu,p,1);
#endif


        /* 
        * Step 8: beta += mu 
        */
        CUBLAS_CALL(cublasSaxpy(cublas_handle, p, &one, d_mu, 1, d_beta, 1));
#ifdef DEBUG
        printf("=After step 8: d_beta == \n");
        d_print_mat(d_beta,p,1);
#endif

        // Extract result
        CUDA_CALL(cudaMemcpy(betas+i*p,d_beta,p*sizeof(float),cudaMemcpyDeviceToHost));
    }

    std::ofstream out;
    out.open("result_beta.txt");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < p; j++){
            out << betas[i*p + j] << "\t";
        }
        out << "\n";
    }
    out.close();
    printf("==========FINISH.\n");
    system("Rscript test_normal.R");

    // Free stuff
    free(X);
    free(lambdaSqInv);
    free(z);
    
    CUDA_CALL(cudaFree(d_X));
    CUDA_CALL(cudaFree(d_XtX));
    CUDA_CALL(cudaFree(d_lambdaSqInv));
    CUDA_CALL(cudaFree(d_z));
    CUDA_CALL(cudaFree(d_beta));
    CUDA_CALL(cudaFree(d_SigmaInv));
    CUDA_CALL(cudaFree(d_mu));

    CUBLAS_CALL(cublasDestroy(cublas_handle));
    CUSOLVER_CALL(cusolverDnDestroy(cusolver_handle));
    CURAND_CALL(curandDestroyGenerator(randGen));

    return 0;
}
