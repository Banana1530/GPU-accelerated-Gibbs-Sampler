
#include "common.h"
#include "paragibbs.h"
#include "test.cu"



int main(int argc, char *argv[]){

    bool want_test = true;
    int  n         = -1;
    int  p         = -1;

    int iters = 10000;
    if(argc > 1){
        n = atoi(argv[1]);
    }
    if(argc > 2){
        p = atoi(argv[2]);
    }
    if(argc > 3){
        iters = atoi(argv[3]);
    }
    if(argc > 4){
        want_test = atoi(argv[4]);
    }

    if(want_test){
        test1();
        test2();
        test3();
        test4();
        test5();
        test6();
        test7();
        test8();
    }


    /*
    *   Run Gibbs smampler
    */
    if(n != -1 && p != -1){
        printf("n=%d, p=%d, iter=%d\n",n,p,iters);
        GpuGibbs g("...","...",n,p);
        g.run(iters);
    }

    /*
    *   Run Bayesian regression on a particular data set
    */

    return 0;
}
