#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define ROOT 0

typedef struct 
{
    int number;
    int count_of_number;
} Hist;

void create_hist_type(MPI_Datatype* hist_type)
{
    MPI_Type_contiguous(2, MPI_INT, hist_type);
    MPI_Type_commit(hist_type);
}

void print_hist(Hist* histogram)
{  
    printf("%d : %d\n",histogram->number, histogram->count_of_number); 
}

int calculate_distinct_nums(int* input, int input_size)
{
    int different_nums = 1;
#pragma omp parallel for shared(input) reduction(+: different_nums)
    for (int i = 0; i < input_size; i++)
    {
        int j = 0;
        for (j = 0; j < i; j++)
        {
            if (input[i] == input[j])
                break;
        }
    if (i == j)
        different_nums++;
    }
    return different_nums;
}

int main(int argc, char *argv[])
{
    //GENERAL INTEGERS NEEDED
    int my_rank, num_procs, input_size, subset_size;
    int* input;
    MPI_Datatype hist_type;
    Hist* histogram;
    
    //MPI INIT
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs); 

    //CREATE NEW MPI TYPE == HIST
    create_hist_type(&hist_type);

    if(my_rank == ROOT)
    {
        //GET THE INPUT
        scanf("%d",&input_size);
        getchar();
    
        input = (int*)malloc(sizeof(int)*input_size); 
        char ch;

        for (int i = 0; i < input_size; i++)
           scanf("%d",&input[i]); 
        
        //CREATE THE HISTOGRAM
        int different_nums = calculate_distinct_nums(input, input_size);
        histogram = (Hist*)malloc(sizeof(Hist)*different_nums);
        
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

    //CALCULATE SIZE OF SUBSET FOR EACH PROC
    subset_size = calculate_distinct_nums(work_arr_nums, num_work_for_each);

    //COMPUTE THE HIST OF EACH SUBSET
    if(my_rank == ROOT)
    {
        Hist* proc_hist_answer = (Hist*)malloc(sizeof(Hist)*subset_size);

        //CREATE ALL DISTINCT NUMBERS
        for (int i = 0; i < num_work_for_each; i++)
        {
            for (int j = 0; j < i; j++)
            {
                if(proc_hist_answer[j].number == work_arr_nums[i])
                    break;
            }
            proc_hist_answer[i].number = work_arr_nums[i];
        }
        
        //COUNT HIST
        int count = 0;
    
        for (int i = 0; i < subset_size; i++)
        {
    #pragma omp parallel for shared(work_arr_nums, proc_hist_answer) reduction(+: count)
            for (int j = 0; j < num_work_for_each; j++)
            {
                if(work_arr_nums[j] == proc_hist_answer[i].number)
                    count++;
            }
            proc_hist_answer[i].count_of_number = count;
        }
        
        
    }
    else
    {

    }
    return 0;
}