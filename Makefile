CC=gcc
CFLAGS=-I.
voicehd: voicehd.o
	$(CC) -o voicehd voicehd.o -O3
