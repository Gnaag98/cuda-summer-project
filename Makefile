profiling_filename = profiling.csv

all: build

run:
	./bin/main

profile:
	nvprof --csv --log-file output/$(profiling_filename) ./bin/main

build-and-run:
	make && ./bin/main

build: bin/main

bin/main: obj/main.o | bin
	nvcc -g -G -std=c++20 -o $@ $+

obj/main.o: main.cu | obj
	nvcc -g -G -std=c++20 -o $@ -c $<

bin:
	mkdir -p $@

obj:
	mkdir -p $@
