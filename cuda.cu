#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include "cudaHeader.h"


__global__ void calculateHistByDevice(int* input, int* histogram)
{
    //CREATE HIST FOR EACH BLOCK
    __shared__ int private_hist[SIZE];

    //CREATE INDEX FOR EACH THREAD IN EACH BLOCK
    int index = threadIdx.x + blockIdx.x * blockDim.x ; 

    //INITIATE THE RESULTS
    private_hist[index] = 0;

    printf("check\n");
    
    //COMPUTE HIST FOR EACH BLOCK
    atomicAdd(&private_hist[input[index]], 1);
    
    //MERGE ALL PRIVATE HISTS INTO OUTPUT
    atomicAdd(&histogram[input[index]], private_hist[index]);
    //atomicAdd(&histogram[input[index]], 1);
    //output[index] += private_hist[index];
}




void calculateHistByCuda(int* input, int size_of_input, int* result)
{
    int num_blocks = size_of_input / NUM_THREADS_PER_BLOCK;
    if(size_of_input % NUM_THREADS_PER_BLOCK != 0)
        num_blocks++;

    /*
    //ALLOCATE DATA TO CUDA MEMORY
    int* cuda_input, *cuda_hist;
    cudaMalloc((void**)&cuda_input, size_of_input);
    cudaMalloc((void**)&cuda_hist, SIZE);
    
    //COPY INPUT INTO DEVICE
    cudaMemcpy(cuda_input, input, size_of_input, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_hist, result, SIZE, cudaMemcpyHostToDevice);
    */
    //LUNCH KERNEL
    //calculateHistByCuda<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(input, result);

    calculateHistByDevice<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(input, result);
    

    //COPY RESULT BACK TO HOST
    //cudaMemcpy(result, cuda_hist, SIZE, cudaMemcpyDeviceToHost);
    
    //FREE
    //cudaFree(cuda_input);
    //cudaFree(cuda_hist);
    
    printf("The histogram from cuda:\n");
    for (int i = 0; i < SIZE; i++)
    {
        if(result[i] != 0)
            printf("%d : %d\n",i, result[i]); 
    }
}