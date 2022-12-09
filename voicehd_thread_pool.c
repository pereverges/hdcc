/*********** voicehd ***********/
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
int TRAIN;
int TEST;
int SIZE = 1;
int P = 50;
int DIMENSIONS = 10240;
int CLASSES = 27;
int S = 128;
typedef float f4si __attribute__ ((vector_size (128)));
int BATCH;
int NUM_BATCH;
int NUM_THREADS = 8;
int SPLIT_SIZE;
int SIZE;
f4si* VALUE;
int VALUE_DIM = 100;
f4si* ID;
int INPUT_DIM = 617;

#define THREAD_NUM 8

typedef struct ThreadList {
	pthread_t thread; // The thread object
	struct ThreadList *next;   // Link to next thread
} ThreadList;

typedef struct Job {
	void (*function)(void *); // The worker function
	void *      args;         // Argument to the function
	struct Job *next;         // Link to next Job
} Job;

typedef struct ThreadPool {
    ThreadList *threads;
    ThreadList *rearThreads;
    uint64_t numThreads;
    volatile uint64_t waitingThreads;
    volatile uint8_t isInitialized;
    pthread_mutex_t queuemutex;
    pthread_mutex_t condmutex;
    pthread_cond_t conditional;
    _Atomic uint8_t run;
    uint64_t threadID;
    uint64_t removeThreads;
    pthread_mutex_t endmutex;
    pthread_cond_t endconditional;
    _Atomic uint64_t jobCount;
    Job *FRONT;
    Job *REAR;
} ThreadPool;

typedef enum Status {
	MEMORY_UNAVAILABLE,
	QUEUE_LOCK_FAILED,
	QUEUE_UNLOCK_FAILED,
	SIGNALLING_FAILED,
	BROADCASTING_FAILED,
	COND_WAIT_FAILED,
	THREAD_CREATION_FAILED,
	POOL_NOT_INITIALIZED,
	POOL_STOPPED,
	INVALID_NUMBER,
	WAIT_ISSUED,
	COMPLETED
} ThreadPoolStatus;

static void *threadExecutor(void *pl) {
	ThreadPool *pool = (ThreadPool *)pl;   // Get the pool
	pthread_mutex_lock(&pool->queuemutex); // Lock the mutex
	++pool->threadID;                      // Get an id
    pthread_mutex_unlock(&pool->queuemutex);
    while(pool->run) {
        pthread_mutex_lock(&pool->queuemutex); // Lock the queue mutex
		if(pool->removeThreads > 0) {
            pthread_mutex_lock(&pool->condmutex);
			pool->numThreads--;
			pthread_mutex_unlock(&pool->condmutex);
			break; // Exit the loop
		}
        Job *presentJob = pool->FRONT;            // Get the first job
		if(presentJob == NULL) {
            pthread_mutex_unlock(&pool->queuemutex); // Unlock the mutex

			pthread_mutex_lock(&pool->condmutex); // Hold the conditional mutex
			pool->waitingThreads++; // Add yourself as a waiti
        if(pool->waitingThreads == pool->numThreads) {
            if(pool->isInitialized) {
                pthread_mutex_lock(&pool->endmutex); // Lock the mutex
                pthread_cond_signal(&pool->endconditional);            // Signal the end
                pthread_mutex_unlock(&pool->endmutex);
            } else {                      // We are initializing the pool}
					pool->isInitialized = 1; // Break the busy wait
            }
        }
        pthread_cond_wait(&pool->conditional, &pool->condmutex); // Idle wait on conditional
			if(pool->waitingThreads >0) // Unregister youself as a waiting thread
				pool->waitingThreads--;
			pthread_mutex_unlock(&pool->condmutex); // Woke up! Release the mutex
        } else { // There is a job in the pool

			pool->FRONT = pool->FRONT->next; // Shift FRONT to right
			pool->jobCount--;                // Decrement the count

			if(pool->FRONT == NULL) // No jobs next
				pool->REAR = NULL;  // Reset the REAR
            pthread_mutex_unlock(&pool->queuemutex); // Unlock the mutex
			presentJob->function(presentJob->args); // Execute the job
			free(presentJob); // Release memory for the job
			}
        }
    if(pool->run) {
        pool->removeThreads--; // Alright, I'm shutting now
        pthread_mutex_unlock(&pool->queuemutex);
    }
    pthread_exit((void *)COMPLETED); // Exit

}


ThreadPoolStatus mt_add_thread(ThreadPool *pool, uint64_t threads) {
	if(pool == NULL) { // Sanity check
		printf("\n[THREADPOOL:ADD:ERROR] Pool is not initialized!");
		return POOL_NOT_INITIALIZED;
	}
	if(!pool->run) {
		printf("\n[THREADPOOL:ADD:ERROR] Pool already stopped!");
		return POOL_STOPPED;
	}

	int temp = 0;
	ThreadPoolStatus rc   = COMPLETED;
	pthread_mutex_lock(&pool->condmutex);
	pool->numThreads +=
	    threads; // Increment the thread count to prevent idle signal
	pthread_mutex_unlock(&pool->condmutex);
	uint64_t i = 0;
	for(i = 0; i < threads; i++) {

		ThreadList *newThread =
		    (ThreadList *)malloc(sizeof(ThreadList)); // Allocate a new thread
		newThread->next = NULL;
		temp = pthread_create(&newThread->thread, NULL, threadExecutor,
		                      (void *)pool); // Start the thread
		if(temp) {
			pthread_mutex_lock(&pool->condmutex);
			pool->numThreads--;
			pthread_mutex_unlock(&pool->condmutex);
			temp = 0;
			rc   = THREAD_CREATION_FAILED;
		} else {
			if(pool->rearThreads == NULL) // This is the first thread
				pool->threads = pool->rearThreads = newThread;
			else // There are threads in the pool
				pool->rearThreads->next = newThread;
			pool->rearThreads = newThread; // This is definitely the last thread
		}
	}
	return rc;
}

ThreadPool *mt_create_pool(uint64_t numThreads) {
	ThreadPool *pool = (ThreadPool *)malloc(
	    sizeof(ThreadPool)); // Allocate memory for the pool
	if(pool == NULL) {       // Oops!
		printf("[THREADPOOL:INIT:ERROR] Unable to allocate memory for the pool!");
		return NULL;
	}

	// Initialize members with default values
	pool->numThreads     = 0;
	pool->FRONT          = NULL;
	pool->REAR           = NULL;
	pool->waitingThreads = 0;
	pool->isInitialized  = 0;
	pool->removeThreads  = 0;
	pool->rearThreads    = NULL;
	pool->threads        = NULL;
	pool->jobCount       = 0;
	pool->threadID       = 0;

	pthread_mutex_init(&pool->queuemutex, NULL); // Initialize queue mutex
	pthread_mutex_init(&pool->condmutex, NULL);  // Initialize idle mutex
	pthread_mutex_init(&pool->endmutex, NULL);   // Initialize end mutex

	pthread_cond_init(&pool->endconditional, NULL); // Initialize end conditional
	pthread_cond_init(&pool->conditional, NULL); // Initialize idle conditional
	pool->run = 1; // Start the pool

	if(numThreads < 1) {
		printf("\n[THREADPOOL:INIT:WARNING] Starting with no threads!");
		pool->isInitialized = 1;
	} else {
		mt_add_thread(pool, numThreads); // Add threads to the pool
	}
	while(!pool->isInitialized)
		; // Busy wait till the pool is initialized
	return pool;
}

ThreadPoolStatus mt_add_job(ThreadPool *pool, void (*func)(void *args),
                              void *      args) {
	if(pool == NULL || !pool->isInitialized) { // Sanity check
		printf("\n[THREADPOOL:EXEC:ERROR] Pool is not initialized!");
		return POOL_NOT_INITIALIZED;
	}
	if(!pool->run) {
		printf(
		    "\n[THREADPOOL:EXEC:ERROR] Trying to add a job in a stopped pool!");
		return POOL_STOPPED;
	}
	if(pool->run == 2) {
		printf("\n[THREADPOOL:EXEC:WARNING] Another thread is waiting for the "
		       "pool "
		       "to complete!");
		return WAIT_ISSUED;
	}

	Job *newJob = (Job *)malloc(sizeof(Job)); // Allocate memory
	if(newJob == NULL) {                      // Who uses 2KB RAM nowadays?
		printf(
		    "\n[THREADPOOL:EXEC:ERROR] Unable to allocate memory for new job!");
		return MEMORY_UNAVAILABLE;
	}

	newJob->function = func; // Initialize the function
	newJob->args     = args; // Initialize the argument
	newJob->next     = NULL; // Reset the link

	pthread_mutex_lock(&pool->queuemutex); // Inserting the job, lock the queue

	if(pool->FRONT == NULL) // This is the first job
		pool->FRONT = pool->REAR = newJob;
	else // There are other jobs
		pool->REAR->next = newJob;
	pool->REAR = newJob; // This is the last job

	pool->jobCount++; // Increment the count

	if(pool->waitingThreads > 0) { // There are some threads sleeping, wake'em up

		pthread_mutex_lock(&pool->condmutex);    // Lock the mutex
		pthread_cond_signal(&pool->conditional); // Signal the conditional
		pthread_mutex_unlock(&pool->condmutex);  // Release the mutex

	}

	pthread_mutex_unlock(&pool->queuemutex); // Finally, release the queue
	return COMPLETED;
}


void mt_destroy_pool(ThreadPool *pool) {
	if(pool == NULL || !pool->isInitialized) { // Sanity check
		printf("\n[THREADPOOL:EXIT:ERROR] Pool is not initialized!");
		return;
	}

	pool->run = 0; // Stop the pool

	pthread_mutex_lock(&pool->condmutex);
	pthread_cond_broadcast(&pool->conditional); // Wake up all idle threads
	pthread_mutex_unlock(&pool->condmutex);

	int rc;

	ThreadList *list = pool->threads, *backup = NULL; // For travsersal

	uint64_t i = 0;
	while(list != NULL) {
		rc = pthread_join(list->thread, NULL); //  Wait for ith thread to join

		backup = list;
		list   = list->next; // Continue

		free(backup); // Free ith thread
		i++;
	}

	// Delete remaining jobs
	while(pool->FRONT != NULL) {
		Job *j      = pool->FRONT;
		pool->FRONT = pool->FRONT->next;
		free(j);
	}

	rc = pthread_cond_destroy(&pool->conditional); // Destroying idle conditional
	rc = pthread_cond_destroy(&pool->endconditional); // Destroying end conditional


	rc = pthread_mutex_destroy(&pool->queuemutex); // Destroying queue lock
	rc = pthread_mutex_destroy(&pool->condmutex);  // Destroying idle lock
	rc = pthread_mutex_destroy(&pool->endmutex);   // Destroying end lock

	free(pool); // Release the pool

}

void mt_join(ThreadPool *pool) {
	if(pool == NULL || !pool->isInitialized) { // Sanity check
		printf("\n[THREADPOOL:WAIT:ERROR] Pool is not initialized!");
		return;
	}
	if(!pool->run) {
		printf("\n[THREADPOOL:WAIT:ERROR] Pool already stopped!");
		return;
	}

	pool->run = 2;

	pthread_mutex_lock(&pool->condmutex);
	/*
	if(pool->numThreads == pool->waitingThreads) {
	printf("\n[THREADPOOL:WAIT:INFO] All threads are already idle!");
		pthread_mutex_unlock(&pool->condmutex);
		pool->run = 1;
		return;
	}
	*/
	pthread_mutex_unlock(&pool->condmutex);

	pthread_mutex_lock(&pool->endmutex); // Lock the mutex
	pthread_cond_wait(&pool->endconditional,&pool->endmutex);    // Wait for end signal
	pthread_mutex_unlock(&pool->endmutex); // Unlock the mutex

	pool->run = 1;
}

uint64_t mt_get_job_count(ThreadPool *pool) {
	return pool->jobCount;
}

uint64_t mt_get_thread_count(ThreadPool *pool) {
	return pool->numThreads;
}

struct arg_struct {
    int split_start;
    float* indices;
    ThreadPool* pool;
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
   int i, j;
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

void encode_fun(void* arg){
    int index = ((struct arg_struct*)arg) -> split_start;
    float* indices = ((struct arg_struct*)arg) -> indices;
    f4si* res = ((struct arg_struct*)arg) -> res;
    ThreadPool* pool = ((struct arg_struct*)arg) -> pool;
    int i, j;
    f4si *aux = calloc(DIMENSIONS,sizeof(int));
    for(i = index; i < SPLIT_SIZE+index; ++i){
        if (index < INPUT_DIM){
            for(j = 0; j < NUM_BATCH; j++){
                //pthread_mutex_lock(&pool->condmutex);
                aux[j] += ID[(NUM_BATCH * i) + j] * (VALUE[(int)indices[i]* NUM_BATCH + j]);
                //pthread_mutex_unlock(&pool->condmutex);

            }
        }
    }
    for(j = 0; j < NUM_BATCH; j++){
        pthread_mutex_lock(&pool->condmutex);
        res[j] += aux[j];
        pthread_mutex_unlock(&pool->condmutex);

    }
}


f4si *encode_function(float* indices, ThreadPool* pool){
    struct arg_struct *args = (struct arg_struct *)malloc(sizeof(struct arg_struct));
    f4si *res = calloc(DIMENSIONS,sizeof(int));
    for (int i = 0; i < NUM_THREADS; i++) {
        struct arg_struct *args = (struct arg_struct *)malloc(sizeof(struct arg_struct));
        args -> split_start = i*SPLIT_SIZE;
        args -> indices = indices;
        args -> res = res;
        args -> pool = pool;
        mt_add_job(pool, &encode_fun, args);
    }
    mt_join(pool);
    /*
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
    */
    return res;
}

float* encoding(float* x, ThreadPool* pool){
    float* enc = (float*)encode_function(x, pool);
    hard_quantize(enc,1);
    return enc;
}



void train_loop(float* train[], float* label[], float* classify, int size, ThreadPool* pool){
    float *res[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        res[i] = (float *)malloc(INPUT_DIM * sizeof(float));
    }
    map_range_clamp(train,TRAIN,INPUT_DIM,0,1,0,VALUE_DIM-1,res);
    int i;
    for(i = 0; i < size; i++){
        float* enc = encoding(res[i], pool);
        update_weight(classify,enc,*(label[i]));
        free(res[i]);
        free(enc);
    }
    normalize(classify);
}

float test_loop(float* test[], float* label[], float* classify, int size, ThreadPool* pool){
    float *res[TEST];
    for (int i = 0; i < TEST; i++){
        res[i] = (float *)calloc(INPUT_DIM, sizeof(float));
    }
    map_range_clamp(test,TEST,INPUT_DIM,0,1,0,VALUE_DIM-1,res);
    int i;
    int correct_pred = 0;
    for(i = 0; i < size; i++){
        float* enc = encoding(res[i], pool);
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
    uint64_t size = 8;
	ThreadPool *pool = mt_create_pool(size);

    BATCH = (int) S/sizeof(float);
    NUM_BATCH = (int) ceil(DIMENSIONS/BATCH);
    SPLIT_SIZE = ceil(INPUT_DIM/NUM_THREADS);
    SIZE = SPLIT_SIZE*NUM_THREADS;


    TRAIN = atoi(argv[1]);
    TEST = atoi(argv[2]);
    float *WEIGHT = weights();

    VALUE = level_hv(VALUE_DIM);
    ID = random_hv(INPUT_DIM);
    float *train_data[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        train_data[i] = (float *)calloc(INPUT_DIM, sizeof(float));
    }

    float *train_labels[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        train_labels[i] = (float *)calloc(1, sizeof(int));
    }

    float *test_data[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        test_data[i] = (float *)calloc(INPUT_DIM, sizeof(float));
    }

    float *test_labels[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        test_labels[i] = (float *)calloc(1, sizeof(int));
    }

    load_data(train_data, argv[3]);
    load_data(test_data, argv[5]);
    load_label(train_labels, argv[4]);
    load_label(test_labels, argv[6]);

    struct timespec begin, end;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    train_loop(train_data, train_labels, WEIGHT,TRAIN,pool);

    float acc = test_loop(test_data,test_labels,WEIGHT,TEST,pool);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = end.tv_sec - begin.tv_sec;
    elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
    printf("%d, %f, %f \n", DIMENSIONS,elapsed, acc);


}
