CC = gcc
CFLAGS = -Wall -O2
LIBS = -lOpenCL -lm

all: grayscale_converter

grayscale_converter: main.c
	$(CC) $(CFLAGS) -o grayscale_converter main.c $(LIBS)

clean:
	rm -f grayscale_converter
