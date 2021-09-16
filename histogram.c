#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "cudaHeader.h"

#define ROOT 0
#define SIZE 256
#define OMP_NUM_THREADS 2


void print_hist(const int* histogram, int size)
{  
    printf("The histogram:\n");
    for (int i = 0; i < size; i++)
    {
        if(histogram[i] != 0)
            printf("%d : %d\n",i, histogram[i]); 
    }
}

int main(int argc, char *argv[])
{
    //GENERAL INTEGERS NEEDED
    int my_rank, num_procs, input_size;
    int* input;
    int histogram[SIZE] = {0};
    
    //MPI INIT
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs); 


    if(my_rank == ROOT)
    {
        //GET THE INPUT
        scanf("%d",&input_size);
        getchar();
    
        input = (int*)malloc(sizeof(int)*input_size); 

        for (int i = 0; i < input_size; i++)
        {
            scanf("%d",&input[i]); 
        }  
    }

    //CAST TO ALL THE INPUT SIZE
    MPI_Bcast(&input_size, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    
    //CALCULATING FOR EACH PROCESS HOW MANY NUMBERS TO CHECK
    int num_work_for_each = input_size / num_procs;
    int extra_work = 0;
    
    if(input_size % num_procs != 0)
        extra_work = input_size % num_procs;

    //BUFFER THAT HOLD A SUBSET OF THE INPUT FOR EACH PROCESS 
    int* work_arr_nums = (int*)malloc(sizeof(int)*num_work_for_each);
    
    //THE ROOT DIVIDES THE WORK BETWEEN THE PROCESSES
    MPI_Scatter(input,num_work_for_each,MPI_INT,work_arr_nums,num_work_for_each,MPI_INT,ROOT,MPI_COMM_WORLD); 

    //COMPUTE THE HIST OF SUBSET
    if(my_rank == ROOT)
    {
        
        //COUNT HIST BY OPENMP
        int private_hist[SIZE] = {0};
#pragma omp parallel for shared(work_arr_nums) reduction(+ : private_hist)
        for (int i = 0; i < num_work_for_each; i++)
        {
            private_hist[work_arr_nums[i]]++;
        }
        
        //RECIEVE THE OTHER HALF OF HISTOGRAM FROM OTHER PROC
        int other_hist[SIZE] = {0};
        MPI_Recv(other_hist, SIZE, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        //MERGE THE TWO HISTS
#pragma omp parallel for shared(private_hist, other_hist) reduction(+ : histogram)
        for (int i = 0; i < SIZE; i++)
        {
            histogram[i] = private_hist[i] + other_hist[i];
        }
        
        //TAKE CARE OF RESIDUALS
        if(extra_work != 0)
        {
            int start = num_procs * num_work_for_each;
            for (int i = start; i < start + extra_work; i++)
            {
                histogram[input[i]]++;
            }
        }

        //OUTPUT
        print_hist(histogram, SIZE);

        //FREE
        free(input);

    }
    else
    {
        
        //SPLIT THE WORK INTO 2
        num_work_for_each /= 2;
        int* work_arr_omp = (int*)calloc(num_work_for_each, sizeof(int));
        int* work_arr_cuda = (int*)calloc(num_work_for_each, sizeof(int));

#pragma omp parallel for shared(work_arr_omp, work_arr_nums)
        for (int i = 0; i < num_work_for_each; i++)
        {
            work_arr_omp[i] = work_arr_nums[i];
        }

        int j = 0;
#pragma omp parallel for shared(work_arr_cuda, work_arr_nums)
        for (int i = num_work_for_each; i < num_work_for_each*2; i++)
        {
            work_arr_cuda[j] = work_arr_nums[i];
            j++;
        }
        
        //////////////////////////CALCULATE HALF WITH OPENMP///////////////////////////
        int omp_hist[SIZE] = {0};
        int** thread_hist;
        int thread_id;
        int num_threads;

#pragma omp parallel
        {
            //SET THE NUMBERS
            thread_id = omp_get_thread_num();
            num_threads = omp_get_num_threads();

            #pragma omp single
            thread_hist = (int**)malloc(sizeof(int*)*num_threads);
        }

        //ALLOCATE THE HISTS OF THE THREADS
 #pragma omp parallel for shared(thread_hist)
        for (int i = 0; i < num_threads; i++)
        {
            thread_hist[i] = (int*)calloc(SIZE, sizeof(int));
        }

        //COUNT HIST BY OPENMP
#pragma omp parallel for shared(thread_hist, work_arr_omp)
        for (int i = 0; i < num_work_for_each; i++)
        {
            thread_hist[thread_id][work_arr_omp[i]]++;
        }

        //UNITE ALL THE COPIES
#pragma omp parallel for shared(omp_hist, thread_hist)
        for (int i = 0; i < num_threads; i++)
        {
            for (int j = 0; j < SIZE; j++)
            {
                omp_hist[j] += thread_hist[i][j];
            }
        }
        //FREE ALL
#pragma omp parallel for shared(thread_hist)
        for (int i = 0; i < num_threads; i++)
        {
            free(thread_hist[i]);
        }

        //////////////////////CALCULATE HALF WITH CUDA////////////////////////////////////////////////
        
        int* cuda_hist = calculateHistByCuda(work_arr_cuda, num_work_for_each);

        //////////////////////////////////////////////////////////////////////////////////////////////
        
        //MERGA CUDA WITH OMP
        int total_hist[SIZE] = {0};

        for (int i = 0; i < SIZE; i++)
        {
            total_hist[i] = omp_hist[i] + cuda_hist[i];
        }
        
        //FREE ALL
        free(work_arr_cuda);
        free(work_arr_omp);

        //SEND THE DATA BACK
        MPI_Send(total_hist, SIZE, MPI_INT, ROOT, 1, MPI_COMM_WORLD); 
        
    }

    //FREE ALL
    free(work_arr_nums);
    MPI_Finalize();
    return 0;
}