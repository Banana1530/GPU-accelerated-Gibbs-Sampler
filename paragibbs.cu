#include "paragibbs.h"

// #define DEBUG
int GpuGibbs::updateBeta(){
    cublasSetStream(    cublas_handle,   beta_stream);
    curandSetStream(    randGen,         beta_stream);
    cusolverDnSetStream(cusolver_handle, beta_stream);
#ifdef DEBUG
    printf("=Before step 2: d_SigmaInv == \n");
    // d_print_mat(d_SigmaInv,p,p);
#endif 
    /* 
    * Step2: 
    * d_SigmaInv = d_XtX
    * d_SigmaInv += d_tauSqInv .* d_lambdaSqInv
    * yj <- alpha * xj + yj
    */
    CUDA_CALL(cudaMemcpyAsync(d_SigmaInv,d_XtX,p*p*sizeof(float),cudaMemcpyDeviceToDevice,beta_stream));


#ifdef DEBUG
    printf("=In step 2: d_lambdaSqInv == \n");
    // d_print_mat(d_lambdaSqInv,1,p);
    printf("d_tauSqInv == \n");
    // d_print_mat(d_tauSqInv,1,1);
    printf("d_SigmaInv == \n");
#endif 
    // cudaDeviceSynchronize();    // FIXED TODO: Don't know why I must synchronize here
                                // Maybe the previous cudaMemcpy is asyn? 
    // add to the diatonal
    CUBLAS_CALL(cublasSaxpy(cublas_handle, p, d_tauSqInv, d_lambdaSqInv, 1, d_SigmaInv, p+1));

#ifdef DEBUG
    printf("=After step 2: d_SigmaInv == \n");
    // d_print_mat(d_SigmaInv,p,p);
#endif 
    /* 
    * Step3: Cholesky decomposition
    * Upper(d_SigmaInv) = R, where d_SigmaInv = R R^T
    */
    CUSOLVER_CALL(cusolverDnSpotrf(cusolver_handle, CUBLAS_FILL_MODE_UPPER,p, d_SigmaInv,
                  p,d_cholWorkspace, cholWorkspaceSize*sizeof(float),devInfo ));

#ifdef DEBUG
    printf("==After step 3: d_SigmaInv == \n");
    // d_print_mat(d_SigmaInv,p,p);
#endif 


    /*
    * Step 4: Generate uniforms
    * beta ~iid N(0,1)
    */
    CURAND_CALL(curandGenerateNormal(randGen, d_beta, p, 0, 1));

#ifdef DEBUG
    printf("==After step 4: d_beta == \n");
    // d_print_mat(d_beta,p,1);
#endif


    /* 
    * Step 5: beta = Rs = solve(upper(SigmaInv,beta))
    */
    CUBLAS_CALL(cublasStrsv(cublas_handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 
                p, d_SigmaInv, p, d_beta, 1));

#ifdef DEBUG
    printf("==After step 5: d_beta == \n");
    // d_print_mat(d_beta,p,1);
#endif


    /* 
    * Step 6: mu = Xt z 
    */

    CUBLAS_CALL(cublasSgemv(cublas_handle,CUBLAS_OP_T,n,p,oneptr,d_X,n,d_z,1,zeroptr,d_mu,1));
    
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
    CUBLAS_CALL(cublasSaxpy(cublas_handle, p, oneptr, d_mu, 1, d_beta, 1));

#ifdef DEBUG
    printf("=After step 8: d_beta == \n");
    d_print_mat(d_beta,p,1);
#endif

    /*
    * Scale beta /= d_sigmaInv,
    */
    // TODO: set grid size
#ifdef DEBUG
    printf("=Before scaling: d_tauSqInv == \n");
    d_print_mat(d_tauSqInv,1,1);
#endif

    shrink_vector <<< block_size(32, p), 32 ,0, beta_stream >>> (d_beta,p,d_tauSqInv);

#ifdef DEBUG
    printf("=After scaling: d_beta == \n");
    d_print_mat(d_beta,p,1);
#endif
    return 0;
}

int GpuGibbs::updateResidual(){
    cublasSetStream(cublas_handle, residual_stream);

    float minusone = -1.0;
    CUDA_CALL(cudaMemcpy(d_residual,d_z,n*sizeof(float),cudaMemcpyDeviceToDevice));
    CUBLAS_CALL(cublasSgemv(cublas_handle,CUBLAS_OP_N,n,p,&minusone,d_X,n,d_beta,1,oneptr,d_residual,1));
    return 0;
}

int GpuGibbs::updateBeta0(){
    return 0;
}

int GpuGibbs::updateLambdaSqInv(){
    thrust::cuda::par.on(lambdaSqInv_stream);
    curandSetStream(randGen_exp, lambdaSqInv_stream);


    float a = 0.5f * *d_tauSqInv * *d_sigmaSqInv;
    thrust::transform(thr_d_nuInv, thr_d_nuInv + p ,thr_d_beta, thr_d_exp_para, x_plus_aysq<float>(a));
    curandGenerateUniform(randGen, d_lambdaSqInv, p);
    trans_unif2exp <<< block_size(32,p),32,0,lambdaSqInv_stream>>> (p,d_lambdaSqInv,d_exp_para);
    return 0;
}

int GpuGibbs::updateNuInv(){
    // exp_para = 1 + lambdaSqInv
    thrust::cuda::par.on(nuInv_stream);
    curandSetStream(randGen_exp, nuInv_stream);

    thrust::constant_iterator<int> ones(1);
    thrust::transform(thr_d_lambdaSqInv, thr_d_lambdaSqInv + p ,ones, thr_d_exp_para, thrust::plus<float>());
    curandGenerateUniform(randGen, d_nuInv, p);
    trans_unif2exp <<< block_size(32,p),32,0,nuInv_stream >>> (p,d_nuInv,d_exp_para);

    return 0;
}

int GpuGibbs::updateSigmaSqInv(){
    thrust::cuda::par.on(sigmaSqInv_stream);
    cublasSetStream(cublas_handle, sigmaSqInv_stream);

    // tauSqInv ~ Gamma( (1+p)/2 , xiInv + <betaSq,lambdaSqInv>/2sigmaSq )
    float resnorm;
    float dotprod;

    CUBLAS_CALL(cublasSnrm2(cublas_handle, n, d_residual, 1, &resnorm));
    // CUBLAS_CALL(cublasSdot (cublas_handle, p, d_lambdaSqInv, 1, d_betaSq, 1, &dotprod));
    dotprod = thrust::inner_product(thr_d_beta,thr_d_beta + p,
                                    thr_d_lambdaSqInv,0.0,thrust::plus<float>(),sqfirst_innerprod<float>());
    
    *d_gamma_alpha = 0.5 * (n + p);
    *d_gamma_beta  = 0.5 * (resnorm * resnorm + *d_tauSqInv * dotprod);
 
    // // WARNING: slow method to sample from gamma dist
    rgamma<<<1, 32, 32* sizeof(float), sigmaSqInv_stream>>>
            (d_states, d_sigmaSqInv, d_gamma_alpha, d_gamma_beta);
    return 0;
}

int GpuGibbs::updateXiInv(){
    // float tmp;
    // CUDA_CALL(cudaMemcpy(&tmp,d_tauSqInv,sizeof(float),cudaMemcpyDeviceToHost));
    curandSetStream(randGen_exp, xiInv_stream);
    *d_gamma_beta = 1.0 + *d_tauSqInv;                      // FIXED TODO: find out why *d_tauSqInv does not work here
                                                            // This should not follow updateBeta() directly without syn.
                                                            // TODO: pin those scalars

    curandGenerateUniform(randGen_exp, d_xiInv, 1);     // TODO: pre-generate it and refill the random numbers asynchly
                                                        // WARNING: choosing PHINIX RNG results in the same 
                                                        // number being generated everytime
    trans_unif2exp <<< 1, block_size(32,1) , 0, xiInv_stream>>> (1, d_xiInv, d_gamma_beta);
                                                        // This overlaps with updatebeta(), so it shouldn't matter.
                                                        // But it is a bad practice to launch a small-size task
    return 0;
}

int GpuGibbs::updateTauSqInv(){
    // tauSqInv ~ Gamma( (1+p)/2 , xiInv + <betaSq,lambdaSqInv>/2sigmaSq )
    thrust::cuda::par.on(tauSqInv_stream);
    float dotprod = thrust::inner_product(thr_d_beta,thr_d_beta + p,
                                    thr_d_lambdaSqInv,0.0,thrust::plus<float>(),sqfirst_innerprod<float>());
    *d_gamma_alpha = 0.5 * (1 + p);
    *d_gamma_beta  = *d_xiInv + 0.5 * *d_sigmaSqInv * dotprod;
    
    // // WARNING: slow method to sample from gamma dist
    rgamma<<<1, 32, 32* sizeof(float), tauSqInv_stream>>>
            (d_states, d_tauSqInv, d_gamma_alpha, d_gamma_beta);
    
    return 0;
}

int GpuGibbs::updateOmegaSq(){return 0;} 



int GpuGibbs::run(int n = 10){
    double iStart = cpuSecond();
    double iElaps;
    for(int i = 0; i < n; i++){
        updateBeta();

        // updateXiInv();
        updateNuInv();
        syncStreams();
        
        updateSigmaSqInv();
        updateTauSqInv();
        syncStreams();

        updateLambdaSqInv();
        syncStreams();
        if(i % 100 == 0){
            iElaps = cpuSecond() - iStart;
            printf("Iter = %d, time = %f, expeted total = %f\n",i,iElaps,iElaps/i*n);
        }
    }

    iElaps = cpuSecond() - iStart;
    printf("Time = %f\n",iElaps);
    return 0;
}

int GpuGibbs::syncStreams(){
    CUDA_CALL(cudaStreamSynchronize(beta_stream));
    CUDA_CALL(cudaStreamSynchronize(xiInv_stream));
    CUDA_CALL(cudaStreamSynchronize(nuInv_stream));
    CUDA_CALL(cudaStreamSynchronize(sigmaSqInv_stream));
    CUDA_CALL(cudaStreamSynchronize(tauSqInv_stream));
    CUDA_CALL(cudaStreamSynchronize(lambdaSqInv_stream));
    CUDA_CALL(cudaStreamSynchronize(omegaSq_stream));
    CUDA_CALL(cudaStreamSynchronize(residual_stream));
    
    return 0;
}