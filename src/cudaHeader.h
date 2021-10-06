#pragma once

#define SIZE 256
#define NUM_THREADS_PER_BLOCK 256

void calculateHistByCuda(int* input, int size_of_input, int* result);
