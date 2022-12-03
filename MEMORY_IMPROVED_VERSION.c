/*********** voicehd ***********/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
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
int DIMENSIONS = 300;
int CLASSES = 27;
int* VALUE;
int VALUE_DIM = 100;
int* ID;
int INPUT_DIM = 617;

float get_rand(float low_bound, float high_bound){
    return (float) ((float)rand() / (float)RAND_MAX) * (high_bound - low_bound) + low_bound;
}

float *random_vector(int size, int dimensions, float low_bound, float high_bound){
   float *arr = (float *)malloc(size * dimensions * sizeof(float));
   int i, j;
   for (i = 0; i < size; i++){
      for (j = 0; j < dimensions; j++){
         *(arr + i*dimensions + j) = get_rand(low_bound, high_bound);
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

void load_label(int* data[], char* path){
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

int *random_hv(int size){
   int *arr = (int *)malloc(size * DIMENSIONS * sizeof(int));
   int i, j;
   for (i = 0; i < size; i++){
      for (j = 0; j < DIMENSIONS; j++){
         *(arr + i*DIMENSIONS + j) = rand() % 100 > P ? 1 : -1;
      }
   }
   return arr;
}

int *level_hv(int levels){
    int levels_per_span = levels-1;
    int span = 1;
    int *span_hv = random_hv(span+1);
    float *threshold_hv = random_vector(span,DIMENSIONS,0,1);
    int *hv = (int *)malloc(levels * DIMENSIONS * sizeof(int));

    int i, j;
    for(i = 0; i < levels; i++){
        float t = 1 - ((float)i / levels_per_span);
        for(j = 0; j < DIMENSIONS; j++){
            if((t > *(threshold_hv + j) || i == 0) && i != levels-1){
                *(hv + i*DIMENSIONS + j) = *(span_hv + j);
            } else {
                *(hv + i*DIMENSIONS + j) = *(span_hv + DIMENSIONS + j);
            }
        }
    }
    free(span_hv);
    free(threshold_hv);
    return hv;
}

int *bind(int *a, int *b, int size){
    int i, j;
    int *arr = (int *)malloc(size * DIMENSIONS * sizeof(int));
    for(i = 0; i < size; ++i){
        for(j = 0; j < DIMENSIONS; j++){
            *(arr + (DIMENSIONS*i) + j ) = *(a + (DIMENSIONS * i) + j) * *(b + (DIMENSIONS *i) + j);
        }
    }
    return arr;
}

int *bundle(int *a, int *b){
    int i;
    int *arr = (int *)malloc(DIMENSIONS * sizeof(int));
    for(i = 0; i < DIMENSIONS; i++){
        *(arr + i) = *(a + i) + *(b + i);
    }
    return arr;
}

int *multiset(int *a, int size){
    int i, j;
    int *arr = (int *)calloc(size * DIMENSIONS, sizeof(int));
    for(i = 0; i < size; i++){
        for(j = 0; j < DIMENSIONS; ++j){
            *(arr + j) += *(a + i*DIMENSIONS + j);
        }
    }
    free(a);
    return arr;
}

float* weights(){
    float *arr = (float *)calloc(CLASSES * DIMENSIONS, sizeof(float));
    return arr;
}

void update_weight(float* weights, int* encoding, int feature){
    int i;
    for(i = 0; i < DIMENSIONS; i++){
        *(weights + feature*DIMENSIONS + i) += (float)*(encoding + i);
    }
}

float* linear(int* m1, float* m2, int s1){
    int i, j, k;
    float *arr = (float *)calloc(s1 * CLASSES, sizeof(float));
    for (i = 0; i < s1; ++i) {
       for (j = 0; j < CLASSES; ++j) {
          for (k = 0; k < DIMENSIONS; ++k) {
             *(arr + i*s1 + j) += (float)*(m1 + i*DIMENSIONS + k) * *(m2 + j*DIMENSIONS + k);
          }
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

int* forward(int* weights, float* indices, int size){
    int *res = (int *)malloc(size * DIMENSIONS * sizeof(int));
    int i, j;
    for(i = 0; i < size; i++){
       int* indx = (weights + (int)*(indices + i));
       for(j = 0; j < DIMENSIONS; j++){
           *(res + i*DIMENSIONS + j) = *(weights + (int)*(indices + i) *DIMENSIONS + j);
       }
    }
    return res;
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

void hard_quantize(int *arr, int size){
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

int* encoding(float* x){
    int* f = forward(VALUE,x,INPUT_DIM);
    int* enc = multiset(bind(ID,f,INPUT_DIM),INPUT_DIM);
    free(f);
    hard_quantize(enc,1);
    return enc;
}

void train_loop(float* train[], int* label[], float* classify, int size){
    float *res[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        res[i] = (float *)malloc(INPUT_DIM * sizeof(float));
    }
    map_range_clamp(train,TRAIN,INPUT_DIM,0,1,0,VALUE_DIM-1,res);
    /*
    for (int i = 0; i < TRAIN; i++){
        free(train[i]);
    }
    */
    int i;
    for(i = 0; i < size; i++){
        int* enc = encoding((res[i]));
        update_weight(classify,enc,*(label[i]));
        free(res[i]);
        free(enc);
    }
    normalize(classify);
}
void print_matrix(int* arr, int rows, int cols){
    int i, j;
    for (i = 0; i < rows; i++) {
      for (j = 0; j < cols; j++) {
         printf("%d ", *(arr + i*cols + j));
      }
      printf("\n");
    }
    printf("\n");
}
float test_loop(float* test[], int* label[], float* classify, int size){
    float *res[TEST];
    for (int i = 0; i < TEST; i++){
        res[i] = (float *)malloc(INPUT_DIM * sizeof(float));
    }
    map_range_clamp(test,TEST,INPUT_DIM,0,1,0,VALUE_DIM-1,res);
    /*
    for (int i = 0; i < TRAIN; i++){
        free(test[i]);
    }
    */
    int i;
    int correct_pred = 0;

    for(i = 0; i < size; i++){
        int* enc = encoding((res[i]));
        float *l = linear(enc,classify,CLASSES);
        free(enc);
        int index = argmax(l);
        if((int)index == (int)*(label[i])){
            correct_pred += 1;
        }
        free(l);
    }

    return correct_pred/(float)TEST;
}

int main(int argc, char **argv) {
    srand(42);
    TRAIN = atoi(argv[1]);
    TEST = atoi(argv[2]);
    float *WEIGHT = weights();

    VALUE = level_hv(VALUE_DIM);
    ID = random_hv(INPUT_DIM);
    float *trainx = (float *)malloc(TRAIN * INPUT_DIM * sizeof(float));
    float *testx = (float *)malloc(TEST * INPUT_DIM * sizeof(float));
    int *trainy = (int *)malloc(TRAIN * 1 * sizeof(int));
    int *testy = (int *)malloc(TEST * 1 * sizeof(int));

    float *train_data[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        train_data[i] = (float *)malloc(INPUT_DIM * sizeof(float));
    }

    int *train_labels[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        train_labels[i] = (int *)malloc(1 * sizeof(int));
    }

    float *test_data[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        test_data[i] = (float *)malloc(INPUT_DIM * sizeof(float));
    }

    int *test_labels[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        test_labels[i] = (int *)malloc(1 * sizeof(int));
    }

    load_data(train_data, argv[3]);
    //load_data(trainx, argv[3]);
    load_data(test_data, argv[5]);
    load_label(train_labels, argv[4]);
    load_label(test_labels, argv[6]);

    train_loop(train_data, train_labels, WEIGHT,TRAIN);
    float acc = test_loop(test_data,test_labels,WEIGHT,TEST);
    printf("acc %f ", acc);
}
