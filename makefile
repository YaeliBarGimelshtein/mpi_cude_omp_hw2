build:
	mpicxx -fopenmp -c histogram.c -o histogram.o
	nvcc -c cuda.cu -o cuda.o 
	mpicxx -o exec histogram.o cuda.o /usr/lib/x86_64-linux-gnu/libcudart_static.a -ldl -lrt -fopenmp
	
clean:
	rm -f *.o exec

run:
	mpiexec -n 2 ./exec <input.txt