#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define ROOT 0

typedef struct hist
{
    int number;
    int count_of_number;
} Hist;

void create_hist_type(MPI_Datatype* hist_type)
{
    MPI_Type_contiguous(2, MPI_INT, hist_type);
    MPI_Type_commit(hist_type);
}

int main(int argc, char *argv[])
{
    //GENERAL INTEGERS NEEDED
    int my_rank, num_procs, input_size;
    int* input;
    MPI_Datatype hist_type;
    Hist* histogram;
    
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
        char ch;

        for (int i = 0; i < input_size; i++)
           scanf("%d",&input[i]); 
        
        //CREATE THE HISTOGRAM
        int different_nums = 0;
#pragma omp parallel for 
        for (int i = 0; i < input_size; i++)
        {
            
        }
        create_hist_type(&hist_type);
        histogram = (Hist*)malloc(sizeof(Hist)*input_size); 

    }
     return 0;
}