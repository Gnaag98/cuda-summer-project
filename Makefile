

all: build

build_and_run:
	make && ./bin/main

build: main

main.o: main.cu
	nvcc -g -G -o $@ -c $<

main: main.o
	nvcc -g -G -o bin/$@ $+
