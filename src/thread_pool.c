#include "thread_pool.h" // API header
#include <pthread.h>   // The thread library
#include <stdio.h>     // Standard output functions in case of errors and debug
#include <stdlib.h>    // Memory management functions

typedef struct ThreadList {
	pthread_t thread;          // The thread object
	struct ThreadList *next;   // Link to next thread
} ThreadList;

typedef struct Job {
	void (*function)(void *); // The worker function
	void * args;              // Argument to the function
	struct Job *next;         // Link to next Job
} Job;

struct ThreadPool {
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
};
#include <sys/syscall.h>

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
	ThreadPoolStatus rc = COMPLETED;
	pthread_mutex_lock(&pool->condmutex);
	pool->numThreads += threads; // Increment the thread count to prevent idle signal
	pthread_mutex_unlock(&pool->condmutex);
	uint64_t i = 0;
	for(i = 0; i < threads; i++) {
		ThreadList *newThread = (ThreadList *)malloc(sizeof(ThreadList)); // Allocate a new thread
		newThread->next = NULL;
		//printf("create pthread %d\n", &newThread->thread);
		temp = pthread_create(&newThread->thread, NULL, threadExecutor, (void *)pool); // Start the thread
		if(temp) {
			pthread_mutex_lock(&pool->condmutex);
			pool->numThreads--;
			pthread_mutex_unlock(&pool->condmutex);
			temp = 0;
			rc = THREAD_CREATION_FAILED;
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

ThreadPoolStatus mt_add_job(ThreadPool *pool, void (*func)(void *args), void *args) {
	if(pool == NULL || !pool->isInitialized) { // Sanity check
		printf("\n[THREADPOOL:EXEC:ERROR] Pool is not initialized!");
		return POOL_NOT_INITIALIZED;
	}
	if(!pool->run) {
		printf("\n[THREADPOOL:EXEC:ERROR] Trying to add a job in a stopped pool!");
		return POOL_STOPPED;
	}
	if(pool->run == 2) {
		printf("\n[THREADPOOL:EXEC:WARNING] Another thread is waiting for the pool to complete!");
		return WAIT_ISSUED;
	}

	Job *newJob = (Job *)malloc(sizeof(Job)); // Allocate memory
	if(newJob == NULL) {                      // Who uses 2KB RAM nowadays?
		printf("\n[THREADPOOL:EXEC:ERROR] Unable to allocate memory for new job!");
		return MEMORY_UNAVAILABLE;
	}

	newJob->function = func; // Initialize the function
	newJob->args = args; // Initialize the argument
	newJob->next = NULL; // Reset the link

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
		list = list->next; // Continue
		free(backup); // Free ith thread
		i++;
	}

	// Delete remaining jobs
	while(pool->FRONT != NULL) {
		Job *j = pool->FRONT;
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

void lock_condition(ThreadPool *pool) {
	pthread_mutex_lock(&pool->condmutex);
}

void unlock_condition(ThreadPool *pool) {
	pthread_mutex_unlock(&pool->condmutex);
}