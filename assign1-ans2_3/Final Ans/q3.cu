#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <errno.h>
#include <omp.h>

#define GET_TIME(x); 	if (clock_gettime(CLOCK_MONOTONIC, &(x)) < 0) \
{ perror("clock_gettime( ):"); exit(EXIT_FAILURE); }
#define MATRIX_DIM 1800
#define MIN_ERROR 0.1
// CUDA related
#define BLOCK_SIZE 32
// PThread related
#define MAX_PTHREADS 8
//Help code for switching between Single Precision and Double Precision
#ifdef DP
typedef double Real;
#else
typedef float Real;
#endif

float elapsed_time_msec(struct timespec *begin, struct timespec *end, long *sec, long *nsec)
{
	if (end->tv_nsec < begin->tv_nsec) {
		*nsec = 1000000000 - (begin->tv_nsec - end->tv_nsec);
		*sec = end->tv_sec - begin->tv_sec -1;
	}
	else {
		*nsec = end->tv_nsec - begin->tv_nsec;
		*sec = end->tv_sec - begin->tv_sec;
	}
	return (float) (*sec) * 1000 + ((float) (*nsec)) / 1000000;
}

__global__ void cuda_simple_mat_mul(Real* A, Real* B, Real* C) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	//check for bounds
	if(row < MATRIX_DIM && col < MATRIX_DIM)
	{
		Real sum = 0.f;
		for (int i = 0; i < MATRIX_DIM; i++)
		sum += A[row * MATRIX_DIM + i] * B[i * MATRIX_DIM + col];
		C[row * MATRIX_DIM + col] = sum;
	}
}

void init_matrix(Real matrix[MATRIX_DIM][MATRIX_DIM])
{
	for(int i = 0; i < MATRIX_DIM; i++)
	{
		for(int j = 0; j < MATRIX_DIM; j++)
		{
			matrix[i][j] = 1 + (Real)rand()/(Real)RAND_MAX;
		}
	}
}

void compare_matrices(Real matrix1[MATRIX_DIM][MATRIX_DIM], Real matrix2[MATRIX_DIM][MATRIX_DIM])
{
	for(int i = 0; i < MATRIX_DIM; i++)
	{
		for(int j = 0; j < MATRIX_DIM; j++)
		{
			if((matrix1[i][j] - matrix2[i][j] > MIN_ERROR) && (matrix1[i][j] - matrix2[i][j] > 0))
			{
				printf("Error i=%d : j=%d mat1=%f mat2=%f\n",i,j,matrix1[i][j], matrix2[i][j]);
				return;
			}
		}
	}
	printf("Matrices Match! \n");
}

void serial_mat_mul(Real A[MATRIX_DIM][MATRIX_DIM], Real B[MATRIX_DIM][MATRIX_DIM], Real C[MATRIX_DIM][MATRIX_DIM])	{
	float sum;
	for (int row = 0; row < MATRIX_DIM; row++){
		for (int col = 0; col < MATRIX_DIM; col++){
			sum = 0.f;
			for (int n = 0; n < MATRIX_DIM; n++){
				sum += A[row][n] * B[n][col];
			}
			C[row][col] = sum;
		}
	}
}

__global__ void cuda_tiled_mat_mul(Real * A, Real * B, Real * C) {
	float CValue = 0;
	int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
	int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	__shared__ Real As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ Real Bs[BLOCK_SIZE][BLOCK_SIZE];

	for (int k = 0; k < (BLOCK_SIZE + MATRIX_DIM - 1)/BLOCK_SIZE; k++) {
		// check ranges for the matrices and check for left out parts where
		//  MATRIX_DIM is not an exact multiplication of tile size(BLOCK_SIZE)
		if (k*BLOCK_SIZE + threadIdx.x < MATRIX_DIM && Row < MATRIX_DIM){
			As[threadIdx.y][threadIdx.x] = A[Row*MATRIX_DIM + k*BLOCK_SIZE + threadIdx.x];
		}
		else{
			As[threadIdx.y][threadIdx.x] = 0.0;
		}
		if (k*BLOCK_SIZE + threadIdx.y < MATRIX_DIM && Col < MATRIX_DIM){
			Bs[threadIdx.y][threadIdx.x] = B[(k*BLOCK_SIZE + threadIdx.y)*MATRIX_DIM + Col];
		}
		else{
			Bs[threadIdx.y][threadIdx.x] = 0.0;
		}
		// Wait till all the threads finish before calculating the results
		__syncthreads();
		for (int n = 0; n < BLOCK_SIZE; ++n)
			CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
		__syncthreads();
	}
	// Calculate the result
	if (Row < MATRIX_DIM && Col < MATRIX_DIM)
		C[((blockIdx.y * blockDim.y + threadIdx.y)*MATRIX_DIM) + (blockIdx.x*blockDim.x)+threadIdx.x] = CValue;
}

int main(int argc, char const *argv[])
{
	if(argc < 2){
		print_usage();
	}
	struct timespec t1, t2;
	long sec, nsec;
	float comp_time;	// in milli seconds
	// Initialize the random seed
	srand(time(NULL));
	// Create the matrices
	static Real A[MATRIX_DIM][MATRIX_DIM];
	static Real B[MATRIX_DIM][MATRIX_DIM];
	static Real C[MATRIX_DIM][MATRIX_DIM];
	static Real serial_C[MATRIX_DIM][MATRIX_DIM];
	// Initialize the matrices
	init_matrix(A);
	init_matrix(B);

	if (0 == strcmp(argv[1], "-s"))
	{
		GET_TIME(t1);
		printf("Serial Mode\n\n");
		// get the serial output
		serial_mat_mul(A,B,serial_C);
		GET_TIME(t2);
		comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
		printf("N = %d: CPU Time(ms) = %.2f \n", MATRIX_DIM, comp_time);
	}
	else if (0 == strcmp(argv[1], "-p"))
	{
		printf("Parallel Mode\n\n");
		int num_of_threads;
		// check whether the given # of threads is valid
		if(argc < 3){
			printf("Need to specify the # of threads at runtime");
			return -1;
		}
		num_of_threads = atoi(argv[2]);
		if(num_of_threads > MAX_PTHREADS){
			printf("Only up to 8 threads can be created\n");
			return -1;
		}
		GET_TIME(t1);
		# pragma omp parallel num_threads(num_of_threads) for private(n,col)
		float sum;
		for (int row = 0; row < MATRIX_DIM; row++){
			for (int col = 0; col < MATRIX_DIM; col++){
				sum = 0.f;
				for (int n = 0; n < MATRIX_DIM; n++){
					sum += A[row][n] * B[n][col];
				}
				C[row][col] = sum;
			}
		}
		GET_TIME(t2);
		comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
		printf("N = %d: OPENMP(%d threads)  Time(ms) = %.2f \n", MATRIX_DIM,num_of_threads, comp_time);
		// if verification is needed
		if((argc == 4) && (0 == strcmp(argv[3], "-v"))){
			GET_TIME(t1);
			// get the serial output
			serial_mat_mul(A,B,serial_C);
			GET_TIME(t1);
			comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
			printf("N = %d: CPU Time(ms) = %.2f \n", MATRIX_DIM, comp_time);
			// Compare the reuslts
			compare_matrices(serial_C,C);
		}
	}
	else if (0 == strcmp(argv[1], "-c"))
	{
		long matrix_size=MATRIX_DIM*MATRIX_DIM*sizeof(Real);
		GET_TIME(t1);
		Real* _A;
		cudaMalloc((void**) &_A, matrix_size);
		Real* _B;
		cudaMalloc((void**) &_B, matrix_size);
		Real* _C;
		cudaMalloc((void**) &_C, matrix_size);
		// copy the matrices to device
		cudaMemcpy(_A, A, matrix_size, cudaMemcpyHostToDevice);
		cudaMemcpy(_B, B, matrix_size, cudaMemcpyHostToDevice);
		// If the tiled mode needs to be enabled
		if (argc > 2 && 0 == strcmp(argv[2], "-t")){
			printf("Cuda Tiled Mode\n");
			// set the grid and block sizes
			dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
			dim3 dimGrid;
			dimGrid.x = (MATRIX_DIM + dimBlock.x - 1)/dimBlock.x;
			dimGrid.y = (MATRIX_DIM + dimBlock.y - 1)/dimBlock.y;
			GET_TIME(t1);
			// execute the workload in the GPU
			cuda_tiled_mat_mul<<<dimGrid , dimBlock>>>(_A,_B,_C);
			// Copy back the result
			cudaMemcpy(C,_C,matrix_size,cudaMemcpyDeviceToHost);
			GET_TIME(t2);
			comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
			printf("N = %d: CUDA Time(ms) = %.2f \n", MATRIX_DIM, comp_time);
			// if verification is needed
			if((argc == 4) && (0 == strcmp(argv[3], "-v"))){
				GET_TIME(t1);
				// get the serial output
				serial_mat_mul(A,B,serial_C);
				GET_TIME(t2);
				comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
				printf("N=%d: CPU Time(ms)=%.2f \n", MATRIX_DIM, comp_time);
				compare_matrices(serial_C,C);
			}
			// free device memory
			cudaFree(_A);
			cudaFree(_B);
			cudaFree(_C);
		}
		else{
			printf("Cuda Mode\n");
			int K=100;
			dim3 threadBlock(BLOCK_SIZE,BLOCK_SIZE);
			dim3 grid(K,K);
			GET_TIME(t1);
			// call the GPU
			cuda_simple_mat_mul<<<grid,threadBlock>>>(_A,_B,_C);
			// Copy back the result
			cudaMemcpy(C,_C,matrix_size,cudaMemcpyDeviceToHost);
			GET_TIME(t2);
			comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
			printf("N = %d: CUDA Time(ms) = %.2f \n", MATRIX_DIM, comp_time);
			// if verification is needed
			if((argc == 3) && (0 == strcmp(argv[2], "-v"))){
				GET_TIME(t1);
				// get the serial output
				serial_mat_mul(A,B,serial_C);
				GET_TIME(t2);
				comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
				printf("N=%d: CPU Time(ms)=%.2f \n", MATRIX_DIM, comp_time);
				compare_matrices(serial_C,C);
			}
			// free device memory
			cudaFree(_A);
			cudaFree(_B);
			cudaFree(_C);
		}
	}
	else{
		printf("Need Specify the Mode of Computation: Serial/Parallel/Cuda");
	}
	return 0;
}
