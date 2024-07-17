profiling_filename = profiling.csv
flags = -std=c++20

all: build

clean:
	rm -rf ./bin
	rm -rf ./obj

run:
	./bin/main

build-debug: flags += -g -G
build-debug: build

build-profile: flags += -lineinfo
build-profile: build

profile:
	nvprof --csv --log-file output/$(profiling_filename) ./bin/main

build-and-run: build run

build: bin/main

bin/main: obj/main.o | bin
	nvcc $(flags) -o $@ $+

obj/main.o: main.cu | obj
	nvcc $(flags) -o $@ -c $<

bin:
	mkdir -p $@

obj:
	mkdir -p $@
