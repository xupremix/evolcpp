.PHONY: all

all: compile run

compile: evol.cpp
	@make -C build --no-print-directory

run: evol.cpp
	@./build/evol
