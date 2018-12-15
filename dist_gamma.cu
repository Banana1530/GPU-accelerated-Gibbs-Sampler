#include "common.h"

__global__ void initialize_state(curandState *states)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(9384, tid, 0, states + tid); // seed, subsequence, offset, state
}

// Ref:
// Terenin, Alexander, Shawfeng Dong, and David Draper. 
// "GPU-accelerated Gibbs sampling: a case study of the Horseshoe Probit model." 
// Statistics and Computing (2018): 1-10.

__global__ void rgamma(curandState *states, float *d_res, float *d_alpha, float *d_beta)
{
    // WARNING: CPU can already give satifactory performance
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // usually blockIdx.x = 0 since we only need one warp
    curandState *state = states + tid;

    extern __shared__ float acc[]; // dynamically allocate shared mem
    __shared__ int success;        //store flag value indicating whether proposal was accepted in shared memory
    float alpha = *d_alpha;
    float beta = *d_beta;

    if (threadIdx.x == 0)
        success = 0; //initialize success

    if (threadIdx.x < blockDim.x && blockIdx.x == 0)
    {
        acc[threadIdx.x] = 0.0f;
        // copy parameters to local memory, can be broadcast to all threads in a block

        // Ref: 
        // Cheng, R. C. H., and G. M. Feast.
        // Some simple gamma variate generators.
        // Applied Statistics (1979): 290-295.
        float a = rsqrtf(2.0f * alpha - 1.0f);
        float b = alpha - 1.3862944f;
        float c = alpha + (1.0f / a);

        int i = 0;
        //perform rejection sampling
        while (success == 0)
        {
          
            
            float u1 = curand_uniform(state);
            float u2 = curand_uniform(state);

            float v = a * logf(u1 / (1.0f - u1));
            float x = alpha * expf(v);

            //perform accept/reject
            if ((b + c * v - x) > logf(u1 * u1 * u2))
            {
                acc[threadIdx.x] = x;
            }
            __syncthreads();

            // find accepted value on thread 0
            if (threadIdx.x == 0)
            {
                for (int j = 0; j < blockDim.x; j++)
                {
                    float gamma_alpha = acc[j];
                    if (gamma_alpha > 0.0f)
                    {
                        *d_res = gamma_alpha / beta;
                        success = 1;
                        break;
                    }
                }
            }
        }
    }
}
