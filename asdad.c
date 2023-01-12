/*********** emgpppp ***********/
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
#define TRAIN 333
#define TEST 143
#define DIMENSIONS 4096
#define CLASSES 5
#define VECTOR_SIZE 128
float *WEIGHT;
typedef float f4si __attribute__ ((vector_size (128)));
#define INPUT_DIM 1024
f4si* SIGNALS;
#define SIGNALS_DIM 21
f4si* CHANNELS;
#define CHANNELS_DIM 1024
#define BATCH 32
#define NUM_BATCH 128
#define NUM_THREADS 4
#define SPLIT_SIZE 256
#define SIZE 1024
#define HIGH 20
int CORRECT_PREDICTIONS;
#define MEMORY_BATCH 200
ThreadPool *pool;
struct DataReader {
    char *path;
    FILE *fp;
    char * line;
    size_t len;
    ssize_t read;
};
struct DataReader* train_data;
struct DataReader* train_labels;
struct DataReader* test_data;
struct DataReader* test_labels;
struct Task {
    float* data;
    int label;
};

struct DataReader* set_load_data(char* path, struct DataReader* data_reader){
    data_reader = (struct DataReader *)calloc(1, sizeof(struct DataReader));
    data_reader -> path = path;
    data_reader -> fp = fopen(path, "r");
    data_reader -> line = NULL;
    data_reader -> len = 0;

    if (data_reader -> fp == NULL)
        exit(EXIT_FAILURE);

    return data_reader;
}

float* load_data_next_line(struct DataReader* data_reader){
    float* data = (float *) calloc(INPUT_DIM, sizeof(float));
    char* token;
    if ((data_reader -> read = getline(&data_reader -> line, &data_reader -> len, data_reader -> fp)) != -1) {
        token = strtok(data_reader -> line, ",");
        for (int i = 0; i < INPUT_DIM; i++){
          *(data + i) = (float) atof(token);
          token = strtok(NULL, ",");
        }
    }
    return data;
}

int load_labels_next_line(struct DataReader* data_reader){
    int label;
    char* token;
    if ((data_reader -> read = getline(&data_reader -> line, &data_reader -> len, data_reader -> fp)) != -1) {
         label = atoi(data_reader -> line);
    }
    return label;
}

void prepare_to_load_data(char **argv){
    train_data = set_load_data(argv[1], train_data);
    train_labels = set_load_data(argv[2], train_labels);
    test_data = set_load_data(argv[3], test_data);
    test_labels = set_load_data(argv[4], test_labels);
}

void close_file(struct DataReader* data_reader){
    fclose(data_reader -> fp );
    if (data_reader -> line)
        free(data_reader -> line);
}

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

void weights(){
    WEIGHT = (float *)calloc(CLASSES * DIMENSIONS, sizeof(float));
}

void update_weight(float* encoding, int feature){
    int i;
    for(i = 0; i < DIMENSIONS; i++){
        *(WEIGHT + feature*DIMENSIONS + i) += (float)*(encoding + i);
    }
}

void update_correct_predictions(){
    lock_condition(pool);
    CORRECT_PREDICTIONS += 1;
    unlock_condition(pool);
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

void map_range_clamp_one(float* arr, float out_max, float* res){
    float in_min = 0;
    float in_max = HIGH;
    float out_min = 0;
    int i, j;
    for (j = 0; j < INPUT_DIM; j++){
        float map_range = round(out_min + (out_max - out_min) * (*(arr + j) - in_min) / (in_max - in_min));
        *(res + j) = min(max(map_range,out_min),out_max);
    }
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

f4si *bind(f4si *a, f4si *b){
    int i, j;
    f4si *enc = (f4si *)calloc(DIMENSIONS * INPUT_DIM, sizeof(int));
    for(i = 0; i < INPUT_DIM; ++i){
        for(j = 0; j < NUM_BATCH; j++){
             enc[(NUM_BATCH * i) + j] = a[(NUM_BATCH * i) + j] * b[NUM_BATCH + j];
        }
    }
    return enc;
}

f4si *bind_forward(f4si *a, f4si *b, float* indices, f4si* enc){
    int i, j;
    for(i = 0; i < INPUT_DIM; ++i){
        for(j = 0; j < NUM_BATCH; j++){
            enc[(NUM_BATCH * i) + j] = a[(NUM_BATCH * i) + j] * b[(int)indices[i]* NUM_BATCH + j];
        }
    }
    return enc;
}


f4si *multiset(f4si *a){
    int i, j;
    for(i = 1; i < INPUT_DIM; i++){
        for(j = 0; j < NUM_BATCH; ++j){
            a[j] += a[(NUM_BATCH * i) + j];
        }
    }
    return a;
}

f4si *multiset_forward(f4si *a, float* indices){
    int i, j;
    for(i = 0; i < INPUT_DIM; i++){
        for(j = 0; j < NUM_BATCH; ++j){
            a[j] += a[(int)indices[i] * i + j];
        }
    }
    return a;
}


 f4si *permute(f4si* arr, int dd, int ini, int fi)
{

    int k, j, i;
    float last;
    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    for (i = ini; i < fi; ++i){
      for (j = 0; j < NUM_BATCH; j++){
         for(k = 0; k < BATCH; k++){
            if ((BATCH*j)+k+dd < ((BATCH*NUM_BATCH))){
                if (k+dd >= BATCH){
                    int num = (k+dd) % BATCH;
                    res[(i-ini)*NUM_BATCH+j+1][num] = arr[i*NUM_BATCH+j][k];

                } else {
                    res[(i-ini)*NUM_BATCH+j][k+dd] = arr[i*NUM_BATCH+j][k];
                }
            } else {
                int num = (k+dd) % BATCH;
                res[(i-ini)*NUM_BATCH+j-1][num] = arr[i*NUM_BATCH+j][k];
            }

         }

      }
    }
    return res;
}

f4si *multiset_bind(f4si *a, f4si *b, f4si* enc){
    int i, j;
    for(i = 0; i < INPUT_DIM; ++i){
        for(j = 0; j < NUM_BATCH; j++){
             enc[j] += a[(NUM_BATCH * i) + j] * b[NUM_BATCH + j];
        }
    }
    return enc;
}

f4si *multiset_bind_forward(f4si *a, f4si *b, float* indices, f4si* enc){
    int i, j;
    for(i = 0; i < INPUT_DIM; ++i){
        for(j = 0; j < NUM_BATCH; j++){
            enc[j] += a[(NUM_BATCH * i) + j] * b[(int)indices[i]* NUM_BATCH + j];
        }
    }
    return enc;
}

f4si *bind_aux(f4si *a, f4si *b, int n){
    int i, j;
    f4si *enc = (f4si *)calloc(DIMENSIONS * n, sizeof(int));
    for(i = 0; i < n; ++i){
        for(j = 0; j < NUM_BATCH; j++){
             enc[(NUM_BATCH * i) + j] = a[(NUM_BATCH * i) + j] * b[(NUM_BATCH * i) + j];
        }
    }
    return enc;
}

f4si *multiset_aux(f4si *a, int n){
    int i, j;
    for(i = 1; i < n; i++){
        for(j = 0; j < NUM_BATCH; ++j){
            a[j] += a[(NUM_BATCH * i) + j];
        }
    }
    return a;
}

f4si* ngram(f4si* arr, int n){
    int i, j,k,a;
    f4si * res = calloc(DIMENSIONS*(INPUT_DIM-(n-1)), sizeof(int));
    f4si * sample = calloc(DIMENSIONS*(INPUT_DIM-(n-1)), sizeof(int));
    res = permute(arr,n-1,0,INPUT_DIM-(n-1));
    for (i = 1; i < n; i++){
        sample = permute(arr,n-i-1,i,INPUT_DIM-(n-1)+i);
        res = bind_aux(res,sample,INPUT_DIM-(n-1));
    }
    return multiset_aux(res,INPUT_DIM-(n-1));
}

void encode_train_task(void* task){
    float* data = ((struct Task*)task) -> data;
    int label = ((struct Task*)task) -> label;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    map_range_clamp_one(data,SIGNALS_DIM-1, indices);
    f4si * enc = calloc(DIMENSIONS*INPUT_DIM, sizeof(int));

    enc = bind_forward(CHANNELS,SIGNALS, indices, enc);

    enc = ngram(enc,3);
    hard_quantize((float*)enc,1);
    update_weight((float*)enc,label);
    free(enc);
    free(data);
    free(indices);
}


void encode_test_task(void* task){
    float* data = ((struct Task*)task) -> data;
    int label = ((struct Task*)task) -> label;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    map_range_clamp_one(data,SIGNALS_DIM-1, indices);
    f4si * enc = calloc(DIMENSIONS*INPUT_DIM, sizeof(int));

    enc = bind_forward(CHANNELS,SIGNALS, indices, enc);
    enc = ngram(enc,3);
    float *l = linear((float*)enc);
    if(argmax(l) == label){
        free(l);
        update_correct_predictions();
    }
    free(enc);
    free(data);
    free(indices);
}

void train_loop(){
    int i;
    for(i = 0; i < TRAIN; i++){
        struct Task *task = (struct Task *)calloc(1,sizeof(struct Task));
        task -> data = load_data_next_line(train_data);
        task -> label = load_labels_next_line(train_labels);
        mt_add_job(pool, &encode_train_task, task);
    }
    mt_join(pool);
    normalize();
}

float test_loop(){
    int i;
    for(i = 0; i < TEST; i++){
        struct Task *task = (struct Task *)calloc(1,sizeof(struct Task));
        task -> data = load_data_next_line(test_data);
        task -> label = load_labels_next_line(test_labels);
        mt_add_job(pool, &encode_test_task, task);
    }
    mt_join(pool);
    return CORRECT_PREDICTIONS/(float)TEST;
}

int main(int argc, char **argv) {

    SIGNALS = level_hv(SIGNALS_DIM);
    CHANNELS = random_hv(CHANNELS_DIM);
	pool = mt_create_pool(NUM_THREADS);
    weights();
    prepare_to_load_data(argv);

    struct timespec begin, end;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &begin);

    train_loop();
    float acc = test_loop();

    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = end.tv_sec - begin.tv_sec;
    elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
    printf("emgpppp,%d,%f,%f\n", DIMENSIONS,elapsed, acc);

}
