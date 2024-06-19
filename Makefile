

all: build

run:
	./bin/main

build_and_run:
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
