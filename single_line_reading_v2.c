/*********** mnist1024 ***********/
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
#define TRAIN 60000
#define TEST 10000
#define DIMENSIONS 1024
#define CLASSES 10
#define VECTOR_SIZE 128
float *WEIGHT;
typedef float f4si __attribute__ ((vector_size (128)));
f4si* POSITION;
#define POSITION_DIM 1000
f4si* VALUE;
#define INPUT_DIM 784
#define BATCH 32
#define NUM_BATCH 32
#define NUM_THREADS 4
#define SPLIT_SIZE 196
#define SIZE 784
float *TRAIN_LABELS[TRAIN];
float *TEST_DATA[TEST];
float *TEST_LABELS[TEST];
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

struct EncodeTask {
    int split_start;
    float* indices;
    f4si *res;
};

struct EncodeTaskTrain {
    int split_start;
    float* data;
    float label;
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

void close_file(struct DataReader* data_reader){
    fclose(data_reader -> fp );
    if (data_reader -> line)
        free(data_reader -> line);
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

float* load_train_data_next_line(){
    float* data = (float *) calloc(INPUT_DIM, sizeof(float));
    char* token;
    if ((train_data -> read = getline(&train_data -> line, &train_data -> len, train_data -> fp)) != -1) {
        token = strtok(train_data -> line, ",");
        for (int i = 0; i < INPUT_DIM; i++){
          *(data + i) = (float) atof(token);
          token = strtok(NULL, ",");
        }
    }
    return data;
}

int load_train_labels_next_line(){
    int label;
    char* token;
    if ((train_labels -> read = getline(&train_labels -> line, &train_labels -> len, train_labels -> fp)) != -1) {
         label = atoi(train_labels -> line);
    }
    return label;
}

void load_data(float* data[], char* path){
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    int count = 0;
    fp = fopen(path, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);
    char* token;
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

void load_test(char **argv){
    for (int i = 0; i < TEST; i++){
        TEST_DATA[i] = (float *) calloc(INPUT_DIM, sizeof(float));
        TEST_LABELS[i] = (float *) calloc(1, sizeof(int));
    }
    load_data(TEST_DATA, argv[3]);
    load_label(TEST_LABELS, argv[4]);
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

void map_range_clamp(float* arr[], int size, float out_max, float* res[]){
   float in_min = 0;
   float in_max = 1;
   float out_min = 0;
   int i, j;
   for (i = 0; i < size; i++){
      for (j = 0; j < INPUT_DIM; j++){
        float map_range = round(out_min + (out_max - out_min) * (*(arr[i] + j) - in_min) / (in_max - in_min));
        *(res[i] + j) = min(max(map_range,out_min),out_max);
      }
      free(arr[i]);
   }
}

void map_range_clamp_one(float* arr, float out_max, float* res){
    float in_min = 0;
    float in_max = 1;
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

void encode_fun(void* task){
    int index = ((struct EncodeTask*)task) -> split_start;
    float* indices = ((struct EncodeTask*)task) -> indices;
    f4si* res = ((struct EncodeTask*)task) -> res;
    int i, j;
    f4si *aux = calloc(DIMENSIONS,sizeof(int));
    for(i = index; i < SPLIT_SIZE+index; ++i){
        if (index < INPUT_DIM){
            for(j = 0; j < NUM_BATCH; j++){
                aux[j] += VALUE[(NUM_BATCH * i) + j] * (POSITION[(int)indices[i]* NUM_BATCH + j]);
            }
        }
    }
    for(j = 0; j < NUM_BATCH; j++){
        lock_condition(pool);
        res[j] += aux[j];
        unlock_condition(pool);
    }
    free(aux);
    free(task);
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

void encode_fun_train(void* task){
    int index = ((struct EncodeTaskTrain*)task) -> split_start;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    float* data = ((struct EncodeTaskTrain*)task) -> data;
    int label = ((struct EncodeTaskTrain*)task) -> label;
    map_range_clamp_one(data,POSITION_DIM-1, indices);
    f4si *res = calloc(DIMENSIONS,sizeof(int));
    int i, j;
    f4si *aux = calloc(DIMENSIONS,sizeof(int));
    for(i = index; i < SPLIT_SIZE+index; ++i){
        if (index < INPUT_DIM){
            for(j = 0; j < NUM_BATCH; j++){
                res[j] += VALUE[(NUM_BATCH * i) + j] * (POSITION[(int)indices[i]* NUM_BATCH + j]);
            }
        }
    }

    for(j = 0; j < NUM_BATCH; j++){
        lock_condition(pool);
        res[j] += aux[j];
        unlock_condition(pool);
    }
    free(aux);
    hard_quantize((float *)res,1);
    update_weight((float *)res, label);
    free(res);
    free(indices);
}

void encode_function_train(float* train_data, int label){
    for (int i = 0; i < NUM_THREADS; i++) {
        struct EncodeTaskTrain *task = (struct EncodeTaskTrain *)calloc(1, sizeof(struct EncodeTaskTrain));
        task -> split_start = i*SPLIT_SIZE;
        task -> data = train_data;
        task -> label = label;
        mt_add_job(pool, &encode_fun_train, task);
    }
}

int b = 200;
// b-2 = 3 // esborrar els ultims tres

void train_loop(){
    int i, j;
    for(i = 0; i < floor(TRAIN/b); i++){
        float *train_d[b];
        int train_l[b];
        for (j = 0; j < b; j++){
            train_d[j] = load_data_next_line(train_data);
            train_l[j] = load_labels_next_line(train_labels);
            encode_function_train(train_d[j], train_l[j]);
        }
        //while(mt_get_job_count(pool) > 20);
        mt_join(pool);
        for (j = 0; j < b; j++){
            free(train_d[j]);
        }
    }
    close_file(train_data);
    close_file(train_labels);
    //mt_join(pool);
    normalize();
}


struct EncodeTaskTest {
    int split_start;
    float* data;
    f4si* res;
};

void encode_fun_test(void* task){
    int index = ((struct EncodeTaskTrain*)task) -> split_start;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    float* data = ((struct EncodeTaskTrain*)task) -> data;
    int label = ((struct EncodeTaskTrain*)task) -> label;
    map_range_clamp_one(data,POSITION_DIM-1, indices);
    f4si *res = calloc(DIMENSIONS,sizeof(int));
    int i, j;
    f4si *aux = calloc(DIMENSIONS,sizeof(int));
    for(i = index; i < SPLIT_SIZE+index; ++i){
        if (index < INPUT_DIM){
            for(j = 0; j < NUM_BATCH; j++){
                res[j] += VALUE[(NUM_BATCH * i) + j] * (POSITION[(int)indices[i]* NUM_BATCH + j]);
            }
        }
    }

    for(j = 0; j < NUM_BATCH; j++){
        lock_condition(pool);
        res[j] += aux[j];
        unlock_condition(pool);
    }
    free(aux);
    hard_quantize((float *)res,1);
    update_weight((float *)res, label);
    free(res);
    free(indices);
}

void encode_function_test(float* train_data, int label){
    for (int i = 0; i < NUM_THREADS; i++) {
        struct EncodeTaskTrain *task = (struct EncodeTaskTrain *)calloc(1, sizeof(struct EncodeTaskTrain));
        task -> split_start = i*SPLIT_SIZE;
        task -> data = train_data;
        task -> label = label;
        mt_add_job(pool, &encode_fun_train, task);
    }
}

float test_loop(){
    int i, j;
    int correct_pred = 0;
    int b = 1;
    for(i = 0; i < floor(TEST/b); i++){
        float *test_d[b];
        int test_l[b];
        float *res[b];
        float *enc[b];
        for (j = 0; j < b; j++){
            res[j] = (float *)calloc(INPUT_DIM, sizeof(float));
            enc[j] = (float *)calloc(DIMENSIONS, sizeof(float));
            test_d[j] = load_data_next_line(test_data);
            test_l[j] = load_labels_next_line(test_labels);
            map_range_clamp_one(test_d[j],POSITION_DIM-1,res[j]);
            enc[j] = encodings(res[j]);
        }
        mt_join(pool);
        for (j = 0; j < b; j++){
            float *l = linear(enc[j]);
            int index = argmax(l);
            if((int)index == (test_l[j])){
                correct_pred += 1;
            }
            free(res[j]);
            free(enc[j]);
            free(test_d[j]);
        }
    }
    close_file(test_labels);
    close_file(test_data);
    return correct_pred/(float)TEST;
}

int main(int argc, char **argv) {
    POSITION = level_hv(POSITION_DIM);
    VALUE = random_hv(INPUT_DIM);
	pool = mt_create_pool(NUM_THREADS);
    weights();
    train_data = set_load_data(argv[1], train_data);
    train_labels = set_load_data(argv[2], train_labels);
    test_data = set_load_data(argv[3], test_data);
    test_labels = set_load_data(argv[4], test_labels);

    struct timespec begin, end;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &begin);

    train_loop();

    //load_test(argv);
    float acc = test_loop();

    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = end.tv_sec - begin.tv_sec;
    elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
    printf("mnist,%d,%f,%f\n", DIMENSIONS,elapsed, acc);

}
