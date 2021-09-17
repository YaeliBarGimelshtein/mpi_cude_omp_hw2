build:
	mpicxx -fopenmp -c histogram.c -o histogram.o
	nvcc -c cuda.cu -o cuda.o 
	mpicxx -fopenmp -o exec  histogram.o cuda.o 
	
clean:
	rm -f *.o exec

run:
	mpiexec -n 2 ./exec <input.txt