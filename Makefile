

all: build

build: main

main.o: main.cu
	nvcc -g -G -o $@ -c $<

main: main.o
	nvcc -g -G -o bin/$@ $+
