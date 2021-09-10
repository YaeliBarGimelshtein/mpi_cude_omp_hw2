build:
	mpicc -c histogram.c 
	mpicc -o exec histogram.o -fopenmp

clean:
	rm -f *.o exec

run:
	mpiexec -n 2 ./exec <input.txt