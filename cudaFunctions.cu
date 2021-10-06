#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <iostream>
#include "cudaHeader.h"


__global__ void calculateHistByDevice(int* input, int input_size, int* histogram)
{
    
    //CREATE HIST FOR EACH BLOCK
    __shared__ int private_hist[SIZE];

    //CREATE INDEX FOR EACH THREAD IN EACH BLOCK
    int index = threadIdx.x + blockIdx.x * blockDim.x ; 

    //INITIATE THE RESULTS
    private_hist[index] = 0;
    
    //COMPUTE HIST FOR EACH BLOCK
    if(index < input_size)
        atomicAdd(&private_hist[input[index]], 1);
    
    //MERGE ALL PRIVATE HISTS INTO OUTPUT
     __syncthreads();
    histogram[index] = private_hist[index];
}




void calculateHistByCuda(int* input, int size_of_input, int* result)
{
    int num_blocks = size_of_input / NUM_THREADS_PER_BLOCK;
    if(size_of_input % NUM_THREADS_PER_BLOCK != 0)
    {
        num_blocks++;
    }
    
    //ALLOCATE DATA TO CUDA MEMORY
    int* cuda_input, *cuda_hist;
    int size_for_cuda_input = sizeof(int) * size_of_input;
    int size_for_cuda_res = sizeof(int) * SIZE;
    cudaMalloc((void**)&cuda_input, size_for_cuda_input);
    cudaMalloc((void**)&cuda_hist, size_for_cuda_res);
    
    //COPY INPUT INTO DEVICE
    cudaMemcpy(cuda_input, input, size_for_cuda_input, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_hist, result, size_for_cuda_res, cudaMemcpyHostToDevice);
    
    //LUNCH KERNEL
    calculateHistByDevice<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(cuda_input, size_of_input, cuda_hist);
    
    //COPY RESULT BACK TO HOST
    cudaMemcpy(result, cuda_hist, size_for_cuda_res, cudaMemcpyDeviceToHost);
    
    //FREE
    cudaFree(cuda_input);
    cudaFree(cuda_hist);
    
}