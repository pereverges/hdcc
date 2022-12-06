CC=gcc
CFLAGS=-I.
mnist64: mnist64.o
	$(CC) -o mnist64 mnist64.o -lm -O3
