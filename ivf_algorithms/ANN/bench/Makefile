include ../../../algorithms/bench/parallelDefsANN
BNCHMRK = ivf

CHECKFILES = $(BNCHMRK)Check.o

COMMON =

INCLUDE = -Icommon

%.o : %.C $(COMMON)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

clean :
	rm -f $(BNCHMRK)Check *.o *.pyc
