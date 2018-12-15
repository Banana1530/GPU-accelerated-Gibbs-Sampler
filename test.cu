#include "common.h"
#include "paragibbs.h"
// TODO:
// Different n and p.
// Change all parameters related to the variable that is being tested.
// Do not hardcode the results. Results should depend on the values in thg gpugibbs object
// What if it keeps generating the same set of random numbers.
// Run MCMC for a few iterations and check it.

// Initialization
void test1(){

    printf("====  TEST 1: Initialization ====\n");
    GpuGibbs g("./test_data/data_X.txt","./test_data/data_z.txt");

    /*
    * Test 1: read data and initialize variables
    */

    // Note matrices are stored column-wise in memory

    // X
    thrust::device_ptr<float> dev_ptr(g.d_X);
    float expect_d_X[12] = {1,0,3,1,  1,12,0,1,  0,0,1,0};
    for(int i = 0; i < 12; i++){
        assert(std::abs(dev_ptr[i] - expect_d_X[i]) < 1e-10);
    }

    // z
    dev_ptr = thrust::device_pointer_cast(g.d_z);
    float expect_d_z[4] = {1,1,1,1};
    for(int i = 0; i < g.n; i++){
        assert(std::abs(dev_ptr[i] - expect_d_z[i]) < 1e-10);
    }

    // XtX
    dev_ptr = thrust::device_pointer_cast(g.d_XtX);
    float expted_d_XtX[9] = {11,2,3,  2,146,0,  3,0,1  };
    for(int i = 0; i < 9; i++){
        assert(std::abs(dev_ptr[i] - expted_d_XtX[i]) < 1e-10);
    }

    // beta initiazed to 0
    dev_ptr = thrust::device_pointer_cast(g.d_beta);
    for(int i = 0; i < g.p; i++){
        assert(std::abs(dev_ptr[i]) < 1e-10);
    }
    assert(std::abs(thrust::device_pointer_cast(g.d_beta0)[0]      - 0) < 1e-10);

    // scalar: sigmaSqInv <- 1, xiInv <- 1, tauSqInv <- 1
    assert(std::abs(thrust::device_pointer_cast(g.d_tauSqInv)[0]   - 1) < 1e-10);
    assert(std::abs(thrust::device_pointer_cast(g.d_sigmaSqInv)[0] - 1) < 1e-10);
    assert(std::abs(thrust::device_pointer_cast(g.d_xiInv)[0]      - 1) < 1e-10);

    // p-dim vector: lambdaSqInv <- 1, nuInv <- 1,
    for(int i = 0; i < g.p; i++){
        assert(std::abs(thrust::device_pointer_cast(g.d_lambdaSqInv)[i] - 1) < 1e-10);
        assert(std::abs(thrust::device_pointer_cast(g.d_nuInv)[i]       - 1) < 1e-10);
    }

    // n-dim vector: residual <- y - X beta = y
    for(int i = 0; i < g.n; i++){
        assert(std::abs(thrust::device_pointer_cast(g.d_residual)[i] - 
                        thrust::device_pointer_cast(g.d_z)[i]) < 1e-10);
    }

    printf("====  PASS TEST 1 ====\n\n");
    }

    // Residual update
    void test2(){

    printf("====  TEST 2: Residual update ====\n");

    GpuGibbs g("./test_data/data_X.txt","./test_data/data_z.txt");

    thrust::device_ptr<float> dev_ptr_beta(g.d_beta);
    thrust::device_ptr<float> dev_ptr_res (g.d_residual);

    // case 1
    for(int i = 0; i < g.p; i++){ dev_ptr_beta[i] = 1;};
    g.updateResidual();
    float expect_res1[4] = {-1  , -11 ,   -3 ,   -1};
    for(int i = 0; i < g.n; i++){
        assert(std::abs(dev_ptr_res[i] - expect_res1[i]) < 1e-10);
    }

    // case 2 
    for(int i = 0; i < g.p; i++){ dev_ptr_beta[i] = i+1;};
    g.updateResidual();
    float expect_res2[4] = {-2 ,  -23,   -5,    -2};
    for(int i = 0; i < g.n; i++){
        assert(std::abs(dev_ptr_res[i] - expect_res2[i]) < 1e-10);
    }

    // case 3
    float test_beta[3] = {2.34,1.14,-0.12};
    for(int i = 0; i < g.p; i++){ dev_ptr_beta[i] = test_beta[i];};
    g.updateResidual();
    float expect_res3[4] = { -2.4800,  -12.6800,   -5.9000,   -2.4800};
    for(int i = 0; i < g.n; i++){
        assert(std::abs(dev_ptr_res[i] - expect_res3[i]) < 1e-6);   // cannot pass the test if set to 1e-7
    }
    printf("==== PASS TEST 2 ====\n\n");
}

// Gamma (single update, sigmaSqInv) dist
void test3(){

  printf("====  TEST 3: Gamma (single update, sigmaSqInv) dist ====\n");

  // case 1
  GpuGibbs g("./test_data/data_X.txt","./test_data/data_z.txt");
 
  int N = 8000;
  float *res;
  res = (float*) malloc(N * sizeof(float));

  // case 1: sigmaSqInv ~ Gamma( 3.5, 2 )
  for(int i = 0; i < N; i++){
      g.updateSigmaSqInv();
      cudaMemcpy(res+i,g.d_sigmaSqInv,sizeof(float),cudaMemcpyDeviceToHost);
  }
  
  std::ofstream out;
  out.open("result_gamma.txt");
  for(int i = 0; i < N; i++){
      out << res[i] << "\n";
  }
  out.close();

  printf("Empirical estimate: alpha = %f, beta = %f\n",*(g.d_gamma_alpha), *(g.d_gamma_beta));

  system("Rscript test_gamma.R");

  system("rm ./result_gamma.txt");

  printf("Expected estimate:\n");
  printf("[1] mean  =  1.75 \n[1] var   =  0.875 \n[1] skew  =  1.069045\n");


  // case 2
  thrust::device_ptr<float> dev_ptr_beta(g.d_beta);
  for(int i = 0; i < g.p; i++) dev_ptr_beta[i] = 1;
 

  // sigmaSqInv ~ Gamma( )
  for(int i = 0; i < N; i++){
      g.updateSigmaSqInv();
      cudaMemcpy(res+i,g.d_sigmaSqInv,sizeof(float),cudaMemcpyDeviceToHost);
  }

  out.open("result_gamma.txt");
  for(int i = 0; i < N; i++){
      out << res[i] << "\n";
  }
  
  printf("Empirical estimate: alpha = %f, beta = %f\n",*(g.d_gamma_alpha), *(g.d_gamma_beta));
  system("Rscript test_gamma.R");

  system("rm ./result_gamma.txt");

  printf("Expected estimate:\n");
  printf("[1] mean  =  1 \n[1] var   =  0.2857143 \n[1] skew  =  1.069045\n");

  printf("==== PLEASE EXAMINE TEST 3 ====\n\n");

  out.close();
  free(res);
}

// TEST 4: Normal dist
void test4(){

  printf("====  TEST 4: Normal dist ====\n");

  GpuGibbs g("./test_data/data_X.txt","./test_data/data_z.txt");
  int N = 5000;
  // g.updateBeta();
  float *betas = (float *) malloc(N * g.p * sizeof(float));

  for(int i = 0; i < N; i++){
      g.updateBeta();
      // WARNING: you cannot call updateBeta concecutively
      // without synchronization
      CUDA_CALL(cudaMemcpy(betas+i*g.p, g.d_beta, g.p*sizeof(float), cudaMemcpyDeviceToHost));
  }

  std::ofstream out;
  out.open("result_beta.txt");
  for(int i = 0; i < N; i++){
      for(int j = 0; j < g.p; j++){
          out << betas[i*g.p + j] << "\t";
      }
      out << "\n";
  }
  out.close();

  printf("Empirical estimate:\n");
  system("Rscript test_normal.R");

  printf("\nExpected estimate:\n");
  printf("var = \n");
  printf(" 0.1338   -0.0018   -0.2007\n-0.0018    0.0068    0.0027\n-0.2007    0.0027    0.8011\n");
  printf("mean = \n");
  printf(" 0.4429   0.0892   -0.1643\n");

  free(betas);
  system("rm ./result_beta.txt");

  printf("==== PLEASE EXAMINE TEST 4 ====\n\n");
}

// Exp (single update, xiInv) dist 
void test5(){
  printf("====  TEST 5: Exp (single update, xiInv) dist ====\n");

  GpuGibbs g("./test_data/data_X.txt","./test_data/data_z.txt");

  int N = 5000;
  float *res;
  res = (float*) malloc(N * sizeof(float));

  // set tauSqInv
  thrust::device_ptr<float> dev_ptr( g.d_tauSqInv );
  dev_ptr[0] = 2.32;

  // case 1: xiInv ~ Exp( 1 + tauSqInv )
  for(int i = 0; i < N; i++){
      g.updateXiInv();
      cudaMemcpy(res+i,g.d_xiInv,sizeof(float),cudaMemcpyDeviceToHost);
  }

  std::ofstream out;
  out.open("result_exp.txt");
  for(int i = 0; i < N; i++){
      out << res[i] << "\n";
  }
  out.close();

  printf("Empirical estimate: beta = %f\n",*(g.d_gamma_beta));

  system("Rscript test_exponential.R");

  system("rm ./result_exp.txt");

  printf("Expected estimate:\n");
  printf("[1] mean  =  %f \n[1] var   =  %f \n[1] skew  =  %f\n",
         1.0f/ *(g.d_gamma_beta),1.0f/ *(g.d_gamma_beta) / *(g.d_gamma_beta), 2.0f);


  free(res);
  printf("==== PLEASE EXAMINE TEST 5 ====\n\n");
}

// Gamma (single update, tauInv) dist 
void test6(){
  printf("====  TEST 6: Gamma (single update, tauInv) dist ====\n");

  GpuGibbs g("./test_data/data_X.txt","./test_data/data_z.txt");

  int N = 10000;
  float *res;
  res = (float*) malloc(N * sizeof(float));

  // case 1
  // set tauSqInv
  thrust::device_ptr<float> dev_ptr( g.d_tauSqInv );
  dev_ptr[0] = 2.32;

  // case 1: tauInv ~ Gamma( )
  for(int i = 0; i < N; i++){
      g.updateTauSqInv();
      cudaMemcpy(res+i,g.d_tauSqInv,sizeof(float),cudaMemcpyDeviceToHost);
  }

  std::ofstream out;
  out.open("result_exp.txt");
  for(int i = 0; i < N; i++){
      out << res[i] << "\n";
  }
  out.close();

  printf("Empirical estimate: alpha = %f, beta = %f\n",*(g.d_gamma_alpha),*(g.d_gamma_beta));

  system("Rscript test_exponential.R");

  system("rm ./result_exp.txt");

  printf("Expected estimate:\n");
  float b = *(g.d_gamma_beta);
  float a = *(g.d_gamma_alpha);
  printf("[1] mean  =  %f \n[1] var   =  %f \n[1] skew  =  %f\n",
         a/b,a/b/b, 2.0f/std::sqrt(a));


  // case 2
  thrust::device_ptr<float> dev_ptr_beta(g.d_beta);
  for(int i = 0; i < g.p; i++) dev_ptr_beta[i] = 1;
  for(int i = 0; i < N; i++){
      g.updateTauSqInv();
      cudaMemcpy(res+i,g.d_tauSqInv,sizeof(float),cudaMemcpyDeviceToHost);
  }
  out.open("result_exp.txt");
  for(int i = 0; i < N; i++){
      out << res[i] << "\n";
  }
  out.close();

  printf("Empirical estimate: alpha = %f, beta = %f\n",*(g.d_gamma_alpha),*(g.d_gamma_beta));

  system("Rscript test_exponential.R");

  system("rm ./result_exp.txt");

  printf("Expected estimate:\n");
  b = *(g.d_gamma_beta);
  a = *(g.d_gamma_alpha);
  printf("[1] mean  =  %f \n[1] var   =  %f \n[1] skew  =  %f\n",
         a/b,a/b/b, 2.0f/std::sqrt(a));

         
  printf("==== PLEASE EXAMINE TEST 6 ====\n\n");
  free(res);
}

// Exp dist (multiple, nuInv)
void test7(){
  printf("====  TEST 7: Exp dist (multiple, nuInv)====\n");

  GpuGibbs g("./test_data/data_X.txt","./test_data/data_z.txt");
  int N = 5000;
  for(int i = 0; i < g.p; i++)
      g.thr_d_lambdaSqInv[i] = i;
  float *nus = (float *) malloc(N * g.p * sizeof(float));

  for(int i = 0; i < N; i++){
      g.updateNuInv();
      CUDA_CALL(cudaMemcpy(nus+i*g.p, g.d_nuInv, g.p*sizeof(float), cudaMemcpyDeviceToHost));
  }

  std::ofstream out;
  out.open("result_beta.txt");
  for(int i = 0; i < N; i++){
      for(int j = 0; j < g.p; j++){
          out << nus[i*g.p + j] << "\t";
      }
      out << "\n";
  }
  out.close();

  printf("Empirical estimate:\n");
  system("Rscript test_normal.R");

  printf("\nExpected estimate:\n");
  printf("var_ii = \n");
  for(int i = 0; i < g.p; i++)
      printf("%f\t",1.0f / g.thr_d_exp_para[i]/ g.thr_d_exp_para[i]);
  printf("\nmean = \n");
  for(int i = 0; i < g.p; i++)
      printf("%f\t",1.0f / g.thr_d_exp_para[i] );

  free(nus);
  system("rm ./result_beta.txt");

  printf("\n==== PLEASE EXAMINE TEST 7 ====\n\n");
}

// Exp dist (multiple, lambdaSqInv)
void test8(){
  printf("====  TEST 8: Exp dist (multiple, lambda_inv)====\n");

  GpuGibbs g("./test_data/data_X.txt","./test_data/data_z.txt");
  int N = 5000;
  for(int i = 0; i < g.p; i++){
      g.thr_d_beta[i] = 2*i+1;
      g.thr_d_nuInv[i] = i+1;
  }
  
  float *lambdas = (float *) malloc(N * g.p * sizeof(float));

  for(int i = 0; i < N; i++){
      g.updateLambdaSqInv();
      CUDA_CALL(cudaMemcpy(lambdas+i*g.p, g.d_lambdaSqInv, g.p*sizeof(float), cudaMemcpyDeviceToHost));
  }

  std::ofstream out;
  out.open("result_beta.txt");
  for(int i = 0; i < N; i++){
      for(int j = 0; j < g.p; j++){
          out << lambdas[i*g.p + j] << "\t";
      }
      out << "\n";
  }
  out.close();

  printf("Empirical estimate:\n");
  system("Rscript test_normal.R");

  printf("\nExpected estimate:\n");
  printf("var_ii = \n");
  for(int i = 0; i < g.p; i++)
      printf("%f\t",1.0f / g.thr_d_exp_para[i]/ g.thr_d_exp_para[i]);
  printf("\nmean = \n");
  for(int i = 0; i < g.p; i++)
      printf("%f\t",1.0f / g.thr_d_exp_para[i] );

  free(lambdas);
  system("rm ./result_beta.txt");

  printf("\n==== PLEASE EXAMINE TEST 8====\n\n");
}
