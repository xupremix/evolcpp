.PHONY: all

all: compile run

compile: evol.cpp
	@make -C build --no-print-directory

run: compile
	@./build/evol

valgrind: compile
	@valgrind ./build/evol
