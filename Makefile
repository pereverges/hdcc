CC=gcc
CFLAGS=-I.
mnist: mnist.o
	$(CC) -o mnist mnist.o -lm -O3
