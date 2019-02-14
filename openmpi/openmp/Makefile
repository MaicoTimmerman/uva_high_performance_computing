PROGNAME = multimult.out
SRCFILES = main.c

RUNARGS:=  100000000 100# vec_len timesteps

IMAGEVIEW = display
CC = gcc

WARNFLAGS = -Wall -Werror-implicit-function-declaration -Wshadow \
		  -Wstrict-prototypes -pedantic-errors
CFLAGS = -std=c99 -ggdb -O2 $(WARNFLAGS) -D_POSIX_C_SOURCE=200112 -fopenmp
LFLAGS = -lm -lrt

# Do some substitution to get a list of .o files from the given .c files.
OBJFILES = $(patsubst %.c,%.o,$(SRCFILES))

.PHONY: all run runlocal clean

all: $(PROGNAME)

$(PROGNAME): $(OBJFILES)
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

%.o: %.c
	$(CC) -c $(CFLAGS) -o $@ $<

run: $(PROGNAME)
# 	prun -v -np 1 $(PROGNAME) $(RUNARGS)
#
# runlocal: $(PROGNAME)
	./$(PROGNAME) 1000 1000 1
	./$(PROGNAME) 1000 10000 1
	./$(PROGNAME) 1000 1000000 1
	./$(PROGNAME) 10000 1000 1
	./$(PROGNAME) 10000 10000 1
	./$(PROGNAME) 10000 100000 1
	./$(PROGNAME) 1000000 1000 1
	./$(PROGNAME) 1000000 10000 1
	./$(PROGNAME) 1000000 100000 1
clean:
	rm -fv $(PROGNAME) $(OBJFILES)
