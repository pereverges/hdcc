CC=gcc
all: thread_pool.c thread_pool.h mnist.c
	$(CC) thread_pool.c mnist.c -lpthread -lm -O3 -o mnist
