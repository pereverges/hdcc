/*********** emgpp ***********/
#include "../src/thread_pool.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif
/*********** CONSTANTS ***********/
#define DIMENSIONS 10240
#define VECTOR_SIZE 128
float *WEIGHT;
typedef float f4si __attribute__ ((vector_size (128)));
#define INPUT_DIM 1024
f4si* SIGNALS;
#define SIGNALS_DIM 21
f4si* CHANNELS;
#define CHANNELS_DIM 1024
#define BATCH 32
#define NUM_BATCH 320
#define NUM_THREADS 4
#define SPLIT_SIZE 256
#define SIZE 1024


 f4si *permute8(f4si* arr, int dd, int ini, int fi)
{

    int k, j, i;
    float n[dd];
    float p[dd];

    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    for (i = ini; i < fi; ++i){
      for (j = 0; j < NUM_BATCH; j++){
       if (j == NUM_BATCH-1){
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],24,25,26,27,28,29,30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23);
           for (k = 0; k < dd; k++){
               res[i*NUM_BATCH+j][k] = p[k];
           }
       } else if (j == 0){
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],24,25,26,27,28,29,30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23);
           for (k = 0; k < dd; k++){
               p[k] = res[i*NUM_BATCH+j][k];
           }
       } else {
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],24,25,26,27,28,29,30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23);
           for (k = 0; k < dd; k++){
               n[k] = res[i*NUM_BATCH+j][k];
               res[i*NUM_BATCH+j][k] = p[k];
               p[k] = n[k];
               n[k] = p[k];
           }
       }
      }
    }
    return res;
}

f4si *permute(f4si* arr, int d, int ini, int fi)
    {
    int i, j, k;
    f4si *res = calloc(DIMENSIONS*(fi-ini), sizeof(int));

    for (i = 0; i < (fi-ini); ++i){
        for (j = 0; j < NUM_BATCH; j++){
            if (j < d){
                res[(i*NUM_BATCH)+j] = arr[(i*NUM_BATCH)+NUM_BATCH-d+j];
            } else {
                res[(i*NUM_BATCH)+j] = arr[(i*NUM_BATCH)+j+d];
            }
        }
    }
    free(arr);

    return res;
}

float *permute_seq_(float* arr, int shift, int ini, int fi){
    float *res = calloc(DIMENSIONS*(fi-ini), sizeof(int));

    for (int i = ini; i < fi; i++){
        memmove(res+((i-ini)*DIMENSIONS)+shift, arr+(i*DIMENSIONS), (DIMENSIONS-shift) * sizeof(*res));
        memmove(res+((i-ini)*DIMENSIONS), arr+(i*DIMENSIONS)+(DIMENSIONS-shift), shift * sizeof(*res));
    }
    free(arr);

    return res;
}

float *permute_seq(float* arr, int d, int ini, int fi)
{
    float *res = calloc(DIMENSIONS*(fi-ini), sizeof(int));

    for (int i = ini; i < fi; i++){
        if (d == 0){
           for (int j = 0; j < DIMENSIONS; j++){
                res[((i-ini)*DIMENSIONS)+j] = arr[(i*DIMENSIONS)+j];
            }
        } else {
           for (int j = 0; j < DIMENSIONS; j++){
                res[((i-ini)*DIMENSIONS)+j+d] = arr[(i*DIMENSIONS)+j];
            }
            for (int j = 0; j < d; j++){
                res[((i-ini)*DIMENSIONS)+j] = arr[(i*DIMENSIONS)+DIMENSIONS-d+j];
            }
        }

    }
    free(arr);
    return res;
}

int main(int argc, char **argv) {
    srand(time(NULL));

    f4si * enc = calloc(CHANNELS_DIM, sizeof(int));

    struct timeval st, et;
    gettimeofday(&st,NULL);

    for (int i = 0; i < 100000; i++){
        enc = permute8(enc,8,0,1);
    }
    gettimeofday(&et,NULL);
    int total = ((et.tv_sec - st.tv_sec) * 1000000) + (et.tv_usec - st.tv_usec);

    printf("Permute shuffle: %d micro seconds\n",total/100000);

    gettimeofday(&st,NULL);

    for (int i = 0; i < 100000; i++){
        enc = permute(enc,8,0,1);
    }
    gettimeofday(&et,NULL);
    total = ((et.tv_sec - st.tv_sec) * 1000000) + (et.tv_usec - st.tv_usec);

    printf("Permute vectorial: %d micro seconds\n",total/100000);

    float * encc = calloc(CHANNELS_DIM, sizeof(int));


    gettimeofday(&st,NULL);

    for (int i = 0; i < 100000; i++){
        encc = permute_seq(encc,8,0,1);
    }
    gettimeofday(&et,NULL);
    total = ((et.tv_sec - st.tv_sec) * 1000000) + (et.tv_usec - st.tv_usec);

    printf("Permute sequential: %d micro seconds\n",total/100000);

}