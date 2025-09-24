include parallelDefsANN
BNCHMRK = neighbors

CHECKFILES = $(BNCHMRK)Check.o

COMMON =

INCLUDE = -Icommon

%.o : %.C $(COMMON)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

timeDistance : timeDistance.C $(COMMON)
	$(CC) $(CFLAGS) $(INCLUDE) -o timeDistance timeDistance.C

# $(BNCHMRK)Check : $(CHECKFILES)
# 	$(CC) $(LFLAGS) -o $@ $(CHECKFILES)

clean :
	rm -f $(BNCHMRK)Check *.o *.pyc
