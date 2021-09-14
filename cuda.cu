#include "cudaHeader.h"

__global__ void calculateHistByCuda(int* input, int* output)
{
    //CREATE HIST FOR EACH BLOCK
    __shared__ int private_hist[SIZE] = {0};

    //CREATE INDEX FOR EACH THREAD IN EACH BLOCK
    int index = threadIdx.x + blockIdx.x * blockDim.x ; 

    //COMPUTE HIST FOR EACH BLOCK
    atomicAdd(&private_hist[input[index]], 1);
    //private_hist[input[index]]++;
    
    //MERGE ALL PRIVATE HISTS INTO OUTPUT
    output[index] += private_hist[index];
}




int* calculateHistByCuda(int* input, int size_of_input)
{
    int result[SIZE] = {0};
    int num_threads_per_block = size_of_input / num_threads_per_block;

    calculateHistByCuda<<<NUM_BLOCKS, num_threads_per_block>>>(input, result);
    return result;
}