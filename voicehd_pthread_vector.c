/*********** voicehd ***********/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif
/*********** CONSTANTS ***********/
int TRAIN;
int TEST;
int SIZE = 1;
int P = 50;
int DIMENSIONS = 10240;
int CLASSES = 27;
int S = 256;
typedef float f4si __attribute__ ((vector_size (256)));
int BATCH;
int NUM_BATCH;
f4si* VALUE;
int VALUE_DIM = 100;
f4si* ID;
int INPUT_DIM = 617;
int NUM_THREADS = 4;
int SPLIT_SIZE;
int SIZE;

struct arg_struct {
    int split_start;
    float* indices;
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

float* weights(){
    float *arr = (float *)calloc(CLASSES * DIMENSIONS, sizeof(float));
    return arr;
}

void update_weight_p(float* weights, float* encoding, int feature){
    int i;
    for(i = 0; i < DIMENSIONS; i++){
         *(weights + feature*DIMENSIONS + i) += (float)*(encoding + i);
    }
}

void update_weight(float* weights, float* encoding, int feature){
    int i;
    for(i = 0; i < DIMENSIONS; i++){
        *(weights + feature*DIMENSIONS + i) += (float)*(encoding + i);
    }
}

float* linear(float* m1, float* m2){
    int j, k;
    float *arr = (float *)calloc(CLASSES, sizeof(float));
    for (j = 0; j < DIMENSIONS; ++j) {
      for (k = 0; k < CLASSES; ++k) {
         *(arr + k) += (float)*(m1 + j) * *(m2 + k*DIMENSIONS + j);
      }
   }
    return arr;
}

float norm2(float* weight,int feature){
   float norm = 0.0;
   int i;
   for (i = 0; i < DIMENSIONS; i++){
      norm += *(weight + feature*DIMENSIONS + i) * *(weight + feature*DIMENSIONS + i);
   }
   return sqrt(norm);
}

void normalize(float* weight){
   float eps = 1e-12;
   int i,j;
   for (i = 0; i < CLASSES; i++){
      float norm = norm2(weight,i);
      for (j = 0; j < DIMENSIONS; j++){
        *(weight + i*DIMENSIONS + j) = *(weight + i*DIMENSIONS + j) / max(norm,eps);
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

float** map_range_clamp(float* arr[], int rows, int cols, float in_min, float in_max, float out_min, float out_max, float* res[]){
   int i, j;
   for (i = 0; i < rows; i++){
      for (j = 0; j < cols; j++){
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


void* routine(void* arg) {
    int index = ((struct arg_struct*)arg) -> split_start;
    float* indices = ((struct arg_struct*)arg) -> indices;
    int i, j;
    f4si *arr = calloc(DIMENSIONS, sizeof(float));
    for(i = index; i < SPLIT_SIZE+index; ++i){
        if (index < INPUT_DIM){
            for(j = 0; j < NUM_BATCH; j++){
                arr[j] += ID[i*NUM_BATCH+j] * (VALUE[(int)indices[i]*NUM_BATCH+j]);
            }
        }
    }
    return arr;
}

f4si *encode_fun(f4si *ID, f4si *VALUE, float* indices, int size){
    SPLIT_SIZE = ceil(size/NUM_THREADS);
    SIZE = SPLIT_SIZE*NUM_THREADS;
    pthread_t th[NUM_THREADS];
    int i;
    for (i = 0; i < NUM_THREADS; i++) {
        struct arg_struct *args = (struct arg_struct *)calloc(1, sizeof(struct arg_struct));
        args -> split_start = i*SPLIT_SIZE;
        args -> indices = indices;
        if (pthread_create(&th[i], NULL, &routine, args) != 0) {
            perror("Failed to create thread");
        }
    }
    int j;
    f4si *res = calloc(DIMENSIONS,sizeof(int));
    for (i = 0; i < NUM_THREADS; i++) {
        f4si* r;
        if (pthread_join(th[i], (void**) &r) != 0) {
            perror("Failed to join thread");
        }
        for(j = 0; j < NUM_BATCH; j++){
            res[j] += r[j];
        }
        free(r);
    }
    return res;
}

void* routine_p(void* arg) {
    int index = ((struct arg_struct*)arg) -> split_start;
    float* indices = ((struct arg_struct*)arg) -> indices;
    int i, j;
    f4si *arr = calloc(DIMENSIONS, sizeof(float));
    for(i = index; i < SPLIT_SIZE+index; ++i){
        if (index < INPUT_DIM){
            for(j = 0; j < NUM_BATCH; j++){
                arr[j] += ID[i*NUM_BATCH+j] * (VALUE[(int)indices[i]*NUM_BATCH+j]);
            }
        }
    }
    return arr;
}

f4si *encode_fun_p(f4si *ID, f4si *VALUE, float* indices, int size){
    SPLIT_SIZE = ceil(size/NUM_THREADS);
    SIZE = SPLIT_SIZE*NUM_THREADS;
    pthread_t th[NUM_THREADS];
    int i;
    for (i = 0; i < NUM_THREADS; i++) {
        struct arg_struct *args = (struct arg_struct *)calloc(1, sizeof(struct arg_struct));
        args -> split_start = i*SPLIT_SIZE;
        args -> indices = indices;
        if (pthread_create(&th[i], NULL, &routine_p, args) != 0) {
            perror("Failed to create thread");
        }
    }
    int j;
    f4si *res = calloc(DIMENSIONS,sizeof(int));
    for (i = 0; i < NUM_THREADS; i++) {
        f4si* r;
        if (pthread_join(th[i], (void**) &r) != 0) {
            perror("Failed to join thread");
        }
        for(j = 0; j < NUM_BATCH; j++){
            res[j] += r[j];
        }
        free(r);
    }
    return res;
}

float* encoding_f(float* x){
    float* enc = (float*)encode_fun_p(ID, VALUE, x ,INPUT_DIM);
    hard_quantize(enc,1);
    return enc;
}

float* encoding(float* x){
    float* enc = (float*)encode_fun(ID, VALUE, x ,INPUT_DIM);
    hard_quantize(enc,1);
    return enc;
}

void train_loop(float* train[], float* label[], float* classify, int size){
    float *res[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        res[i] = (float *)calloc(INPUT_DIM, sizeof(float));
    }
    map_range_clamp(train,TRAIN,INPUT_DIM,0,1,0,VALUE_DIM-1,res);
    int i;
    for(i = 0; i < size; i++){
        float* enc;
        if (i == 10){
            enc = encoding_f(res[i]);
        } else {
            enc = encoding(res[i]);
        }
        update_weight(classify,enc,(int)*(label[i]));
        free(res[i]);
        free(enc);
    }
    normalize(classify);
}

float test_loop(float* test[], float* label[], float* classify, int size){
    float *res[TEST];
    for (int i = 0; i < TEST; i++){
        res[i] = (float *)calloc(INPUT_DIM, sizeof(float));
    }
    map_range_clamp(test,TEST,INPUT_DIM,0,1,0,VALUE_DIM-1,res);
    int i;
    int correct_pred = 0;
    for(i = 0; i < size; i++){
        float* enc = encoding(res[i]);
        float *l = linear(enc,classify);
        int index = argmax(l);
        if((int)index == (int)*(label[i])){
            correct_pred += 1;
        }
        free(l);
        free(res[i]);
        free(enc);
    }
    return correct_pred/(float)TEST;
}


int main(int argc, char **argv) {
    srand(42);
    BATCH = (int) S/sizeof(float);
    NUM_BATCH = (int) ceil(DIMENSIONS/BATCH);

    TRAIN = atoi(argv[1]);
    TEST = atoi(argv[2]);
    float *WEIGHT = weights();

    VALUE = level_hv(VALUE_DIM);
    f4si a = VALUE[0]*VALUE[0];
    ID = random_hv(INPUT_DIM);
    float *train_data[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        train_data[i] = (float *)calloc(INPUT_DIM, sizeof(float));
    }

    float *train_labels[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        train_labels[i] = (float *)calloc(1, sizeof(float));
    }

    float *test_data[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        test_data[i] = (float *)calloc(INPUT_DIM, sizeof(float));
    }

    float *test_labels[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        test_labels[i] = (float *)calloc(1, sizeof(float));
    }

    load_data(train_data, argv[3]);
    load_data(test_data, argv[5]);
    load_label(train_labels, argv[4]);
    load_label(test_labels, argv[6]);

    struct timespec begin, end;
    double elapsed;

    clock_gettime(CLOCK_MONOTONIC, &begin);
    printf("Start\n");
    train_loop(train_data, train_labels, WEIGHT,TRAIN);

    float acc = test_loop(test_data,test_labels,WEIGHT,TEST);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = end.tv_sec - begin.tv_sec;
    elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
    printf("Time: %f\n",elapsed);
    printf("Accuracy: %f \n", acc);

}