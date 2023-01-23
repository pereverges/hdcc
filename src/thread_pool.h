#include <stdint.h> // Standard integer

typedef struct ThreadPool ThreadPool;

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

ThreadPool *mt_create_pool(uint64_t);

void mt_join(ThreadPool *);

void mt_destroy_pool(ThreadPool *);

ThreadPoolStatus mt_add_job(ThreadPool *, void (*func)(void *), void *);

ThreadPoolStatus mt_add_thread(ThreadPool *, uint64_t);

uint64_t mt_get_job_count(ThreadPool *pool);

uint64_t mt_get_thread_count(ThreadPool *);

void lock_condition(ThreadPool *pool);

void unlock_condition(ThreadPool *pool);