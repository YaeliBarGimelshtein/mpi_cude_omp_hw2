#include "cudaHeader.h"

__global__ void calculateHistByCuda(int* input, int* histogram)
{
    //CREATE HIST FOR EACH BLOCK
    __shared__ int private_hist[SIZE] = {0};

    //CREATE INDEX FOR EACH THREAD IN EACH BLOCK
    int index = threadIdx.x + blockIdx.x * blockDim.x ; 

    //COMPUTE HIST FOR EACH BLOCK
    atomicAdd(&private_hist[input[index]], 1);
    //private_hist[input[index]]++;
    
    //MERGE ALL PRIVATE HISTS INTO OUTPUT
    atomicAdd(&histogram[index], 1);
    //output[index] += private_hist[index];
}




int* calculateHistByCuda(int* input, int size_of_input)
{
    int result[SIZE] = {0};
    int num_blocks = size_of_input / NUM_THREADS_PER_BLOCK;
    if(size_of_input % NUM_THREADS_PER_BLOCK != 0)
        num_blocks++;

    calculateHistByCuda<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(input, result);
    return result;
}