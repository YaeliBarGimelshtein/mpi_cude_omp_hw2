#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define ROOT 0
#define SIZE 256


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

    //COMPUTE THE HIST OF EACH SUBSET
    if(my_rank == ROOT)
    {
        //COUNT HIST BY OPENMP
        int private_hist[SIZE] = {0};
#pragma omp parallel for shared(work_arr_nums) reduction(+ : private_hist)
        for (int i = 0; i < num_work_for_each; i++)
        {
            private_hist[work_arr_nums[i]]++;
        }
        
        /*
        //RECIEVE THE OTHER HALF OF HISTOGRAM FROM OTHER PROC
        int other_hist[SIZE] = {0};
        MPI_Recv(other_hist, SIZE, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        //MERGE THE TWO HISTS
#pragma omp parallel for shared(private_hist, other_hist)
        for (int i = 0; i < SIZE; i++)
        {
            histogram[i] = private_hist[i] + other_hist[i];
        }

        //OUTPUT
        print_hist(histogram, SIZE);
        */
        print_hist(private_hist, SIZE);
    }
    else
    {
        //CALCULATE HALF WITH OPENMP
#pragma omp parallel
        {
            int num_threads = omp_get_num_threads();
            int* hist_for_each_thread[num_threads];
            printf("from process 1 , num threads is %d\n", num_threads);
        }
        
        /*
#pragma omp parallel for shared(hist_for_each_thread)
        for (int i = 0; i < num_threads; i++)
        {
            hist_for_each_thread[i] = (int*)calloc(SIZE, sizeof(int));
        }
        

#pragma omp parallel for shared(work_arr_nums,hist_for_each_thread)
        for (int i = 0; i < num_work_for_each; i++)
        {
            
        }
        
        //CALCULATE HALF WITH CUDA
        */

    }
    return 0;
}