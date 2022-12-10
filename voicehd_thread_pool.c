/*********** voicehd ***********/
#include "thread_pool.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif
/*********** CONSTANTS ***********/


#define TRAIN 6238
#define TEST 1559

#define DIMENSIONS 10240
#define CLASSES 27
#define VECTOR_SIZE 128
#define INPUT_DIM 617

typedef float f4si __attribute__ ((vector_size (VECTOR_SIZE)));
//int BATCH = 32;
#define BATCH 32 // BATCH = (int) VECTOR_SIZE/sizeof(float);
#define NUM_BATCH 320 // NUM_BATCH = (int) ceil(DIMENSIONS/BATCH);
#define NUM_THREADS 4

#define SPLIT_SIZE 154 // SPLIT_SIZE = ceil(INPUT_DIM/NUM_THREADS);
#define SIZE 616 // SIZE = SPLIT_SIZE*NUM_THREADS;

f4si* VALUE;
#define VALUE_DIM 100
f4si* ID;
float *WEIGHT;
ThreadPool *pool;

float *TRAIN_DATA[TRAIN];
float *TRAIN_LABELS[TRAIN];
float *TEST_DATA[TEST];
float *TEST_LABELS[TEST];

struct EncodeTask {
    int split_start;
    float* indices;
    f4si *res;
};

float get_rand(float low_bound, float high_bound){
    return (float) ((float)rand() / (float)RAND_MAX) * (high_bound - low_bound) + low_bound;
}

f4si *random_vector(int size, float low_bound, float high_bound){
   f4si *arr = calloc(size * DIMENSIONS, sizeof(float));
   int i, j, k;
   for (i = 0; i < size; i++){
      for (j = 0; j < NUM_BATCH; j++){
         for(k = 0; k < BATCH; k++){
            arr[i*NUM_BATCH+j][k] = get_rand(low_bound, high_bound);
         }
      }
   }
   return arr;
}

void load_data(float* data[], char* path){
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    fp = fopen(path, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);
    char* token;
    int count = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
        token = strtok(line, ",");
        for (int i = 0; i < INPUT_DIM; i++){
          *(data[count] + i) = (float) atof(token);
          token = strtok(NULL, ",");
        }
        count += 1;
    }
    fclose(fp);
    if (line)
        free(line);
}

void load_label(float* data[], char* path){
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    fp = fopen(path, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);
    char* token;
    int count = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
        *(data[count]) = atoi(line);
        count += 1;
    }
    fclose(fp);
    if (line)
        free(line);
}

f4si *random_hv(int size){
   f4si *arr = calloc(size * DIMENSIONS, sizeof(float));
   int P = 50;
   int i, j, k;
   for (i = 0; i < size; i++){
      for (j = 0; j < NUM_BATCH; j++){
         for(k = 0; k < BATCH; k++){
            arr[i*NUM_BATCH+j][k] = rand() % 100 > P ? 1 : -1;
         }
      }
   }
   return arr;
}

f4si *level_hv(int levels){
    int levels_per_span = levels-1;
    int span = 1;
    f4si *span_hv = random_hv(span+1);
    f4si *threshold_hv = random_vector(span,0,1);
    f4si *hv = calloc(levels * DIMENSIONS, sizeof(float));
    int i, j, k;
    for(i = 0; i < levels; i++){
        float t = 1 - ((float)i / levels_per_span);
        for(j = 0; j < NUM_BATCH; j++){
            for(k = 0; k < BATCH; k++){
                if((t > threshold_hv[j][k] || i == 0) && i != levels-1){
                    hv[i*NUM_BATCH+j][k] = span_hv[0*NUM_BATCH+j][k];
                } else {
                    hv[i*NUM_BATCH+j][k] = span_hv[1*NUM_BATCH+j][k];
                }
             }
        }
    }
    return hv;
}

void weights(){
    WEIGHT = (float *)calloc(CLASSES * DIMENSIONS, sizeof(float));
}

void update_weight(float* encoding, int feature){
    int i;
    for(i = 0; i < DIMENSIONS; i++){
        *(WEIGHT + feature*DIMENSIONS + i) += (float)*(encoding + i);
    }
}

float* linear(float* m){
    int j, k;
    float *arr = (float *)calloc(CLASSES, sizeof(float));
    for (j = 0; j < DIMENSIONS; ++j) {
      for (k = 0; k < CLASSES; ++k) {
         *(arr + k) += (float)*(m + j) * *(WEIGHT + k*DIMENSIONS + j);
      }
   }
    return arr;
}

float norm2(int feature){
   float norm = 0.0;
   int i;
   for (i = 0; i < DIMENSIONS; i++){
      norm += *(WEIGHT + feature*DIMENSIONS + i) * *(WEIGHT + feature*DIMENSIONS + i);
   }
   return sqrt(norm);
}

void normalize(){
   float eps = 1e-12;
   int i, j;
   for (i = 0; i < CLASSES; i++){
      float norm = norm2(i);
      for (j = 0; j < DIMENSIONS; j++){
        *(WEIGHT + i*DIMENSIONS + j) = *(WEIGHT + i*DIMENSIONS + j) / max(norm,eps);
      }
   }
}

int argmax(float* classify){
   int i;
   int max_i = 0;
   float max = 0;
   for (i = 0; i < CLASSES; i++){
       if(*(classify + i) > max){
           max = *(classify + i);
           max_i = i;
       }
   }
   return max_i;
}

float** map_range_clamp(float* arr[], int size, float* res[]){
   float in_min = 0;
   float in_max = 1;
   float out_min = 0;
   float out_max = VALUE_DIM-1;
   int i, j;
   for (i = 0; i < size; i++){
      for (j = 0; j < INPUT_DIM; j++){
        float map_range = round(out_min + (out_max - out_min) * (*(arr[i] + j) - in_min) / (in_max - in_min));
        *(res[i] + j) = min(max(map_range,out_min),out_max);
      }
      free(arr[i]);
   }
   return res;
}

void hard_quantize(float *arr, int size){
   int i, j;
   for (i = 0; i < size; i++){
      for (j = 0; j < DIMENSIONS; j++){
        int value = *(arr + i*DIMENSIONS + j);
        if (value > 0){
          *(arr + i*DIMENSIONS + j) = 1.0;
        } else {
            *(arr + i*DIMENSIONS + j) = -1.0;
        }
      }
   }
}

void encode_fun(void* task){
    int index = ((struct EncodeTask*)task) -> split_start;
    float* indices = ((struct EncodeTask*)task) -> indices;
    f4si* res = ((struct EncodeTask*)task) -> res;
    int i, j;
    f4si *aux = calloc(DIMENSIONS,sizeof(int));
    for(i = index; i < SPLIT_SIZE+index; ++i){
        if (index < INPUT_DIM){
            for(j = 0; j < NUM_BATCH; j++){
                aux[j] += ID[(NUM_BATCH * i) + j] * (VALUE[(int)indices[i]* NUM_BATCH + j]);
            }
        }
    }
    for(j = 0; j < NUM_BATCH; j++){
        lock_condition(pool);
        res[j] += aux[j];
        unlock_condition(pool);
    }
}


f4si *encode_function(float* indices){
    struct EncodeTask *task = (struct EncodeTask *)malloc(sizeof(struct EncodeTask));
    f4si *res = calloc(DIMENSIONS,sizeof(int));
    for (int i = 0; i < NUM_THREADS; i++) {
        struct EncodeTask *task = (struct EncodeTask *)malloc(sizeof(struct EncodeTask));
        task -> split_start = i*SPLIT_SIZE;
        task -> indices = indices;
        task -> res = res;
        mt_add_job(pool, &encode_fun, task);
    }
    return res;
}

float* encodings(float* x){
    float* enc = (float*)encode_function(x);
    hard_quantize(enc,1);
    return enc;
}

void train_loop(){
    float *res[TRAIN];
    float *enc[TRAIN];
    int i;
    for (i = 0; i < TRAIN; i++){
        res[i] = (float *)calloc(INPUT_DIM, sizeof(float));
        enc[i] = (float *)calloc(DIMENSIONS, sizeof(float));
    }
    map_range_clamp(TRAIN_DATA,TRAIN,res);
    for(i = 0; i < TRAIN; i++){
        enc[i] = encodings(res[i]);
    }
    mt_join(pool);
    for(i = 0; i < TRAIN; i++){
        update_weight(enc[i],*(TRAIN_LABELS[i]));
        free(res[i]);
        free(enc[i]);
    }
    normalize();
}

float test_loop(){
    float *res[TEST];
    float *enc[TEST];
    int i;
    for (i = 0; i < TEST; i++){
        res[i] = (float *)calloc(INPUT_DIM, sizeof(float));
        enc[i] = (float *)calloc(DIMENSIONS, sizeof(float));
    }
    map_range_clamp(TEST_DATA,TEST,res);
    int correct_pred = 0;
    for(i = 0; i < TEST; i++){
        enc[i] = encodings(res[i]);
    }
    mt_join(pool);
    for(i = 0; i < TEST; i++){
        float *l = linear(enc[i]);
        int index = argmax(l);
        if((int)index == (int)*(TEST_LABELS[i])){
            correct_pred += 1;
        }
        free(l);
        free(res[i]);
        free(enc[i]);
    }
    return correct_pred/(float)TEST;
}

void load_dataset(char **argv){
    for (int i = 0; i < TRAIN; i++){
        TRAIN_DATA[i] = (float *)calloc(INPUT_DIM, sizeof(float));
        TRAIN_LABELS[i] = (float *)calloc(1, sizeof(int));
    }

    for (int i = 0; i < TEST; i++){
        TEST_DATA[i] = (float *)calloc(INPUT_DIM, sizeof(float));
        TEST_LABELS[i] = (float *)calloc(1, sizeof(int));
    }

    load_data(TRAIN_DATA, argv[3]);
    load_data(TEST_DATA, argv[5]);
    load_label(TRAIN_LABELS, argv[4]);
    load_label(TEST_LABELS, argv[6]);
}

int main(int argc, char **argv) {
    VALUE = level_hv(VALUE_DIM);
    ID = random_hv(INPUT_DIM);

	pool = mt_create_pool(NUM_THREADS);
    weights();

    load_dataset(argv);

    struct timespec begin, end;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    train_loop();

    float acc = test_loop();
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = end.tv_sec - begin.tv_sec;
    elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
    printf("%d, %f, %f \n", DIMENSIONS,elapsed, acc);
}
