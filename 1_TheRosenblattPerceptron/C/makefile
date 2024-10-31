#-----------------------------------------------------------------#
#					Makefile for UNIX Systems | WSL | Ubuntu Version
#										using a GNU C Compiler
#-----------------------------------------------------------------#

CC=gcc
CFLAGS=-g -D__USE_FIXED_PROTOTYPES__ -ansi -std=c11

#
# Compiler flags:
#				-g			-- Enable debugging
#				-Wall		-- Turn on all arnings (not used since it gives aways the bug in this program)
#				-D__USE_FIXED_PROTOTYPES__
#								-- Force the compiler to use the correct headers
#				-amso		-- Dont use GNU extensions. Stick to ANSI C.
#

perceptron: perceptron.c 
	$(CC) $(CFLAGS) -o ./bin/perceptron perceptron.c -lm

clean:
	rm -f ./bin/perceptron