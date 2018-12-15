#ifndef _PARAGIBBS_H
#define _PARAGIBBS_H
#include "common.h"


class GpuGibbs{
public:
    float *oneptr;
    float *zeroptr;

    int n;
    int p;

    int *d_n;
    int *d_p;

    // p-dim vector
    float *d_beta;
    thrust::device_ptr<float> thr_d_beta;         // TODO: maybe put to the class def
    float *d_lambdaSqInv;
    thrust::device_ptr<float> thr_d_lambdaSqInv;
    float *d_nuInv;
    thrust::device_ptr<float> thr_d_nuInv;
    float *d_exp_para;
    thrust::device_ptr<float> thr_d_exp_para;

    // scalar
    float *d_beta0;         // no sampling for now
    float *d_tauSqInv;
    float *d_xiInv;
    float *d_sigmaSqInv;

    // n-dim vector
    float *d_omegaSq;       // no smapling for now
    
    // n * p matrix
    float *d_X;
    // n vector
    float *d_z;             // no smapling
    // p * p matrix
    float *d_XtX;

    // working space for multi-normal sampling
    // p * p
    float *d_SigmaInv;
    // p
    float *d_mu;
    // n
    float *d_residual;
    thrust::device_ptr<float> thr_d_residual;
    // p
    float *d_betaSq;
    // scalar, sigmaInv update
    float *d_gamma_alpha;
    float *d_gamma_beta;
  

    int cholWorkspaceSize;
    int *devInfo;
    float *d_cholWorkspace;
    
    // working space for random seed states sampling
    

    // handler
    cublasHandle_t     cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    curandGenerator_t  randGen;
    curandGenerator_t  randGen_exp;
    curandState        *d_states;      // random seeds used in samplin from a gamma distribution, only need 32 states

    // Streams
    cudaStream_t beta_stream;
    cudaStream_t xiInv_stream;
    cudaStream_t nuInv_stream;
    cudaStream_t sigmaSqInv_stream;
    cudaStream_t tauSqInv_stream;
    cudaStream_t lambdaSqInv_stream;
    cudaStream_t omegaSq_stream;
    cudaStream_t residual_stream;


    // initialize a device space with a constant
    int set_device_matrix(float value,float * dst,int row,int col){
        float *tmp = (float *)malloc(row*col*sizeof(float));
        for(int i = 0; i < row * col; i++) tmp[i] = value;
        CUDA_CALL(cudaMemcpy(dst,tmp,row*col*sizeof(float),cudaMemcpyHostToDevice));
        free(tmp);
        return 0;
    }
    
public:
    int updateBeta();
    int updateBeta0();

    int updateLambdaSqInv();
    int updateNuInv();

    int updateTauSqInv();
    int updateXiInv();
    
    int updateSigmaSqInv();
    int updateOmegaSq();

    int updateResidual();

    int syncStreams();

    int run(int);

    GpuGibbs(const char *X_filename, const char *y_filename, int in_n=-1, int in_p=-1){

        oneptr   = (float *) malloc(sizeof(float));
        zeroptr  = (float *) malloc(sizeof(float));
        *oneptr  = 1.0;
        *zeroptr = 0.0;


        CUSOLVER_CALL(cusolverDnCreate(&cusolver_handle));
        CUBLAS_CALL(cublasCreate(&cublas_handle));
        CURAND_CALL(curandCreateGenerator(&randGen,             
                    CURAND_RNG_PSEUDO_PHILOX4_32_10));      // Host: generate multi-Normal. 
                                                            // NOTE 1: Choice of rng could affect results
                                                            //         e.g., variance does not match theoretical result.
                                                            // NOTE 2: URAND_RNG_PSEUDO_DEFAULT will prompt 
                                                            //         CURAND_STATUS_LENGTH_NOT_MULTIPLE

        CURAND_CALL(curandCreateGenerator(&randGen_exp,         // Host: generate uniforms
                    CURAND_RNG_PSEUDO_DEFAULT));                // If CURAND_RNG_PSEUDO_PHILOX4_32_10 is used,
                                                                // it gives the same number everytime (?)
                                                                // in updateXiInv()

        // this is used to generate gamma dist, using device API      
        CUDA_CALL(cudaMalloc((void **)&d_states, sizeof(curandState) * 1 * 32));
        initialize_state <<< 1, 32>>> (d_states);


        // Read data to CPU memory
        float *h_X;
        int n_z;
        float *h_z;
        if(in_p == -1 && in_n == -1){

            h_X = read_mat_col_maj(X_filename,&n,&p);
            h_z = read_vec(y_filename ,&n_z); 
            assert(n_z == n);
        }
        else{
            p = in_p;
            n = in_n;
        }
        printf("[INFO]: Initialize a GpuGibbs object, data size: n = %d, p = %d.\n",n,p);

        // Allocate GPU memory
        CUDA_CALL(cudaMalloc((void **)&d_n, sizeof(float))); 
        CUDA_CALL(cudaMalloc((void **)&d_p, sizeof(float))); 

        CUDA_CALL(cudaMalloc((void **)&d_beta,        p*sizeof(float))); 
        CUDA_CALL(cudaMalloc((void **)&d_lambdaSqInv, p*sizeof(float))); 
        CUDA_CALL(cudaMalloc((void **)&d_nuInv,       p*sizeof(float))); 
        CUDA_CALL(cudaMalloc((void **)&d_exp_para,    p*sizeof(float))); 

        CUDA_CALL(cudaMalloc(       (void **)&d_beta0,      sizeof(float))); 
        CUDA_CALL(cudaMallocManaged((void **)&d_tauSqInv,   sizeof(float)));    // Unified memory, TODO: pin it!
        CUDA_CALL(cudaMallocManaged((void **)&d_xiInv,      sizeof(float)));    // Unified memory
        CUDA_CALL(cudaMallocManaged((void **)&d_sigmaSqInv, sizeof(float)));    // Unified memory
 
        CUDA_CALL(cudaMalloc((void **)&d_omegaSq, n*sizeof(float))); 

        CUDA_CALL(cudaMalloc((void **)&d_X,   n*p*sizeof(float)));
        CUDA_CALL(cudaMalloc((void **)&d_z,   n*sizeof(float))); 
        CUDA_CALL(cudaMalloc((void **)&d_XtX, p*p*sizeof(float)));

        CUDA_CALL(cudaMalloc((void **)&d_SigmaInv, p*p*sizeof(float)));
        CUDA_CALL(cudaMalloc((void **)&d_mu,       p*sizeof(float)));
        CUDA_CALL(cudaMalloc((void **)&d_residual, n*sizeof(float)));
        CUDA_CALL(cudaMalloc((void **)&d_betaSq,   p*sizeof(float)));

        CUDA_CALL(cudaMallocManaged((void**)&d_gamma_alpha,sizeof(float)));
        CUDA_CALL(cudaMallocManaged((void**)&d_gamma_beta, sizeof(float)));

        // Initialize working space for normal sampling
        CUDA_CALL(cudaMalloc((void **)&devInfo,sizeof(int)));
        CUSOLVER_CALL(cusolverDnSpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_UPPER, 
                      p, d_SigmaInv, 1, &cholWorkspaceSize));
        CUDA_CALL(cudaMalloc((void **)&d_cholWorkspace, cholWorkspaceSize*sizeof(float)));


        // Copy data matrix/vector
        CUDA_CALL(cudaMemcpy(d_n,&n,sizeof(int),cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_p,&p,sizeof(int),cudaMemcpyHostToDevice));

        set_device_matrix(0,d_beta,        p,1);
        set_device_matrix(1,d_lambdaSqInv, p,1);
        set_device_matrix(1,d_nuInv,       p,1);
        // d_exp_para need not initialize

        CUDA_CALL(cudaMemcpy(d_beta0,      zeroptr, sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_tauSqInv,   oneptr,  sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_xiInv,      oneptr,  sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_sigmaSqInv, oneptr,  sizeof(float),cudaMemcpyHostToDevice));

        set_device_matrix(1,d_omegaSq,n,1);

        // d_X and d_z
        if(in_p == -1 && in_n == -1) {
            CUBLAS_CALL(cublasSetMatrix(n, p, sizeof(float), h_X, n, d_X, n));
            CUDA_CALL  (cudaMemcpy(d_z,h_z,n*sizeof(float),cudaMemcpyHostToDevice));
        }else{
            printf("[INFO]: Start generating data\n");
            curandGenerator_t  randGen;
            CURAND_CALL(curandCreateGenerator(&randGen,             
                CURAND_RNG_PSEUDO_PHILOX4_32_10));


            // Generate X
            curandGenerateNormal(randGen, d_X, p * n, 0, 1);
            // Generate z 
            curandGenerateNormal(randGen, d_z, n, 0, 1);


            CURAND_CALL(curandDestroyGenerator(randGen));
            printf("[INFO]: Finish generating data\n");
        }
        CUBLAS_CALL(cublasSgemm(cublas_handle,CUBLAS_OP_T,CUBLAS_OP_N,p,p,n,oneptr,
                    d_X,n,d_X,n,zeroptr,d_XtX,p));

        // d_SigmaInv, d_mu do not need initialization
        set_device_matrix(0,d_SigmaInv,p,p);
        // d_cholWorkspace does not need init

        // thrust pointers
        thr_d_beta        = thrust::device_pointer_cast(d_beta);         // TODO: pin it!
        thr_d_lambdaSqInv = thrust::device_pointer_cast(d_lambdaSqInv);
        thr_d_nuInv       = thrust::device_pointer_cast(d_nuInv);
        thr_d_residual    = thrust::device_pointer_cast(d_residual);
        thr_d_exp_para    = thrust::device_pointer_cast(d_exp_para);


        // create streams
        CUDA_CALL(cudaStreamCreate(&beta_stream));
        CUDA_CALL(cudaStreamCreate(&xiInv_stream));
        CUDA_CALL(cudaStreamCreate(&nuInv_stream));
        CUDA_CALL(cudaStreamCreate(&sigmaSqInv_stream));
        CUDA_CALL(cudaStreamCreate(&tauSqInv_stream));
        CUDA_CALL(cudaStreamCreate(&lambdaSqInv_stream));
        CUDA_CALL(cudaStreamCreate(&omegaSq_stream));
        CUDA_CALL(cudaStreamCreate(&residual_stream));

        updateResidual();

        cudaDeviceSynchronize();    // Important!
        printf("[INFO]: Successful initialization\n");
    }


    ~GpuGibbs(){
        free(oneptr);
        free(zeroptr);

        CUDA_CALL(cudaFree(d_n));
        CUDA_CALL(cudaFree(d_p));

        CUDA_CALL(cudaFree(d_beta));
        CUDA_CALL(cudaFree(d_lambdaSqInv));
        CUDA_CALL(cudaFree(d_nuInv));
        CUDA_CALL(cudaFree(d_exp_para));

        CUDA_CALL(cudaFree(d_beta0));
        CUDA_CALL(cudaFree(d_tauSqInv));
        CUDA_CALL(cudaFree(d_xiInv));
        CUDA_CALL(cudaFree(d_SigmaInv));

        CUDA_CALL(cudaFree(d_omegaSq));
        
        CUDA_CALL(cudaFree(d_X));
        CUDA_CALL(cudaFree(d_z));
        CUDA_CALL(cudaFree(d_XtX));
        
        CUDA_CALL(cudaFree(d_mu));
        CUDA_CALL(cudaFree(d_residual));
        CUDA_CALL(cudaFree(d_betaSq));
        CUDA_CALL(cudaFree(d_gamma_alpha));
        CUDA_CALL(cudaFree(d_gamma_beta));

        CUDA_CALL(cudaFree(devInfo));
        CUDA_CALL(cudaFree(d_cholWorkspace));


        CUBLAS_CALL(cublasDestroy(cublas_handle));
        CUSOLVER_CALL(cusolverDnDestroy(cusolver_handle));
        CURAND_CALL(curandDestroyGenerator(randGen));
        CURAND_CALL(curandDestroyGenerator(randGen_exp));

        // destroy streams
        CUDA_CALL(cudaStreamDestroy(beta_stream));
        CUDA_CALL(cudaStreamDestroy(xiInv_stream));
        CUDA_CALL(cudaStreamDestroy(nuInv_stream));
        CUDA_CALL(cudaStreamDestroy(sigmaSqInv_stream));
        CUDA_CALL(cudaStreamDestroy(tauSqInv_stream));
        CUDA_CALL(cudaStreamDestroy(lambdaSqInv_stream));
        CUDA_CALL(cudaStreamDestroy(omegaSq_stream));
        CUDA_CALL(cudaStreamDestroy(residual_stream));
    }
};

#endif // _PARAGIBBS_H
