build:
	mpicxx -fopenmp -c histogram.c -o histogram.o
	mpicxx -fopenmp -o exec  histogram.o 
	
clean:
	rm -f *.o exec

run:
	mpiexec -n 2 ./exec <input.txt