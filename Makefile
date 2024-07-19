flags = -std=c++20 -arch=sm_60 --expt-relaxed-constexpr

all: build

build: bin/main_atomic bin/main_shared

build-debug: flags += -g -G
build-debug: build

build-profile: flags += -lineinfo
build-profile: build

run:
	bin/main_atomic
	bin/main_shared

build-and-run: build run

clean:
	rm -rf ./bin
	rm -rf ./obj

bin/main_atomic: obj/main_atomic.o | bin
	nvcc $(flags) -o $@ $+

bin/main_shared: obj/main_shared.o | bin
	nvcc $(flags) -o $@ $+

obj/%.o: %.cu common.hpp | obj
	nvcc $(flags) -o $@ -c $<

bin:
	mkdir -p $@

obj:
	mkdir -p $@
