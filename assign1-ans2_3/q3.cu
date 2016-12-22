// How to use?
// ===========
// Single precision Serial mode:
// 
// clear;rm a.out; nvcc -O3 q3.cu ;./a.out -s
// 
// Single precision PThreads mode: 
// 
// clear;rm a.out; nvcc -O3 q3.cu ;./a.out -p <num of threads>
// 
// Single precision Cuda simple calculation mode:
// 
// clear;rm a.out; nvcc -O3 q3.cu ;./a.out -c
// 
// Single precision Cuda tiled mode:
// 
// clear;rm a.out; nvcc -O3 q3.cu ;./a.out -c -t
// 
// To use with Double precision, use the -D DP option when compiling. Also for
// CUDA based Double Precision calculation, a -arch sm_20 flag is recommended
// to turn off the "Double is not supported. Demoting to float" warning.
// 
// Add a -v flag at the end when running the code if verification is needed.
// ex - clear;rm a.out; nvcc -O3 q3.cu ;./a.out -c -t -v
// 

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>

#define GET_TIME(x); 	if (clock_gettime(CLOCK_MONOTONIC, &(x)) < 0) \
{ perror("clock_gettime( ):"); exit(EXIT_FAILURE); }	


#define MATRIX_DIM 1800

#define MIN_ERROR 0.1

// CUDA related
#define BLOCK_SIZE 32

// PThread related
#define MAX_PTHREADS 8

//Code to check for GPU errors
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code),\
			file, line);
		if (abort) exit(code);
	}
}

//Help code for switching between Single Precision and Double Precision
#ifdef DP
typedef double Real;
#else
typedef float Real;
#endif

/**
 * Measures the time differences
 * @param  begin begin time
 * @param  end   end time
 * @param  sec   resulting time in seconds
 * @param  nsec  resulting time in nano-seconds
 * @return       the time taken
 */
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
 static unsigned long inKB(unsigned long bytes)

 { return bytes/1024; }



 static unsigned long inMB(unsigned long bytes)

 { return bytes/(1024*1024); }


/**
 * Used to print memory states in the GPU
 */
 static void printStats()

 {

 	size_t free, total;

 	CUresult res = cuMemGetInfo(&free, &total);

 	if(res != CUDA_SUCCESS){
 		printf("!!!! cuMemGetInfo failed! (status = %x)", res);
 		return;

 	}

 	printf("---------------------------------------------------------------\n");

 	printf("^^^^ Free : %lu bytes (%lu KB) (%lu MB)\n", free, inKB(free), \
 		inMB(free));

 	printf("^^^^ Total: %lu bytes (%lu KB) (%lu MB)\n", total, inKB(total), \
 		inMB(total));

 	printf("^^^^ %f%% free, %f%% used\n", 100.0*free/(double)total, \
 		100.0*(total - free)/(double)total);
 	printf("---------------------------------------------------------------\n");

 }

/**
 * Carries out a simple square matrix multiplication where each thread
 * calculates a single item in the resulting matrix.
 * @param A First matrix
 * @param B Second matrix
 * @param C Results matrix
 */
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

/**
 * Initializes the given matrix to a set of float/Double values between 1-2
 */
 void init_matrix(Real matrix[MATRIX_DIM][MATRIX_DIM])
 {
 	for(int i=0; i < MATRIX_DIM; i++)
 	{
 		for(int j=0; j < MATRIX_DIM; j++)
 		{
 			matrix[i][j] = 1 + (Real)rand()/(Real)RAND_MAX;
 		}
 	}
 }

/**
 * Prints the given matrix to the stdout
 */
 void print_matrix(Real matrix[MATRIX_DIM][MATRIX_DIM])
 {

 	for(int i = 0; i < MATRIX_DIM; i++)
 	{
 		printf("[");
 			for(int j  = 0; j < MATRIX_DIM; j++)
 			{
		#ifdef DP
 				printf("%20.18f ", matrix[i][j]);
    	#else
 				printf("%f ", matrix[i][j]);
    	#endif

 				
 			}
 			printf("] \n");
 		}
 		printf("\n");
 	}

/**
 * Compares the given two matrices.
 */
 void compare_matrices(Real matrix1[MATRIX_DIM][MATRIX_DIM],\
 	Real matrix2[MATRIX_DIM][MATRIX_DIM])
 {
 	for(int i = 0; i < MATRIX_DIM; i++)
 	{
 		for(int j = 0; j < MATRIX_DIM; j++)
 		{
 			if((matrix1[i][j] - matrix2[i][j] > MIN_ERROR) &&
 				(matrix1[i][j] - matrix2[i][j] > 0))
 			{
 				printf("Error i=%d : j=%d mat1=%f mat2=%f\n",i,j,\
 					matrix1[i][j], matrix2[i][j]);
 				return;
 			}
 		}
 	}

 	printf("Matrices Match! \n");
 } 
/**
 * carries out a serial matrix multiplication
 */
 void serial_mat_mul(Real A[MATRIX_DIM][MATRIX_DIM], \
 	Real B[MATRIX_DIM][MATRIX_DIM], Real C[MATRIX_DIM][MATRIX_DIM])	{
 	float sum;
 	for (int row=0; row<MATRIX_DIM; row++){
 		for (int col=0; col<MATRIX_DIM; col++){
 			sum = 0.f;
 			for (int n=0; n<MATRIX_DIM; n++){
 				sum += A[row][n]*B[n][col];
 			}
 			C[row][col] = sum;
 		}
 	}
 }

/**
 * Shows the usage of the program.
 */
 void print_usage(){
 	printf("Wrong usage!\n");
 }

/**
 * Does a matrix multiplication using the "tiled" approach in the GPU
 * @param A First matrix
 * @param B Second matrix
 * @param C Results matrix
 */
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

 			As[threadIdx.y][threadIdx.x] = A[Row*MATRIX_DIM + \
 				k*BLOCK_SIZE + threadIdx.x];
 		}  
 		else{

 			As[threadIdx.y][threadIdx.x] = 0.0;
 		}                                                   

 		if (k*BLOCK_SIZE + threadIdx.y < MATRIX_DIM && Col < MATRIX_DIM){
 			
 			Bs[threadIdx.y][threadIdx.x] = B[(k*BLOCK_SIZE + \
 				threadIdx.y)*MATRIX_DIM + Col];
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
 		C[((blockIdx.y * blockDim.y + threadIdx.y)*MATRIX_DIM)+\
 			(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;

 }

 //struct for parameter passing between pthread calls
 struct pthread_arg_struct {
 	int tid;
 	int total_threads;
 	Real (*A)[MATRIX_DIM];
 	Real (*B)[MATRIX_DIM];
 	Real (*C)[MATRIX_DIM];
 };

/**
 * PThread code for assigning tasks to pthreads
 * @param arguments an instance of pthread_arg_struct
 */
 void* pthread_mat_mul(void* arguments)
 {
 	struct pthread_arg_struct *args = (struct pthread_arg_struct *)arguments;
 	int total_threads = args -> total_threads;
	int tid = args -> tid;       //obtain the value of thread id
	Real (*A)[MATRIX_DIM]=args -> A;
	Real (*B)[MATRIX_DIM]=args -> B;

	// get the workload for one thread
	int chunk_size=MATRIX_DIM/total_threads;

	// check for the row ranges the thread needs to calculate 
	int min_row = chunk_size * tid;
	int max_row = (min_row+chunk_size-1) < MATRIX_DIM ? (min_row+chunk_size-1) : MATRIX_DIM;

	float sum=0.f;
	// loop the matrix entries that belongs to this thread
	for(;min_row<=max_row;min_row++){
		for(int col=0;col<MATRIX_DIM;col++){
			for (int n=0; n<MATRIX_DIM; n++){
				sum += A[min_row][n]*B[n][col];
			}
			args->C[min_row][col] = sum;
			sum=0;

		}
	}
	

	pthread_exit((void*)0);	
}

int main(int argc, char const *argv[])
{

	if(argc<2){
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
 	// print_matrix(A);
 	// print_matrix(B);


	if (0 == strcmp(argv[1], "-s"))
	{
		GET_TIME(t1);

		printf("serial mode\n\n");
		// get the serial output
		serial_mat_mul(A,B,serial_C);

		GET_TIME(t2);

		comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
		printf("N=%d: CPU Time(ms)=%.2f \n", MATRIX_DIM, comp_time);

	}
	else if (0 == strcmp(argv[1], "-p"))
	{
		printf("pthread mode\n\n");

		int num_of_threads;
		// check whether the given # of threads is valid
		if(argc <3){
			print_usage();
			return -1;
		}
		num_of_threads=atoi(argv[2]);
		if(num_of_threads>MAX_PTHREADS){
			printf("[ERROR-PTHREADS] - Only up to 8 threads can be created\n");
			return -1;
		}

		pthread_t threads[num_of_threads];
		int rc;
		long t;
		void *status;

		GET_TIME(t1);

		//initialize the threads
		for(t=0;t<num_of_threads;t++){
			struct pthread_arg_struct* args=(\
				struct pthread_arg_struct*)malloc(sizeof *args);

			args->total_threads=num_of_threads;
			args->tid=t;
			args-> A=A;
			args-> B=B;
			args-> C=C;

			rc = pthread_create(&threads[t], NULL, pthread_mat_mul,(void *)args);
			if (rc){
				printf("ERROR; return code from pthread_create() is %d\n", rc);
				exit(-1);
			}
		}

		//join the threads
		for(t=0;t<num_of_threads;t++){
			pthread_join(threads[t], &status);
		}

		GET_TIME(t2);
		comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
		printf("N=%d: PThreads(%d threads)  Time(ms)=%.2f \n", MATRIX_DIM,num_of_threads, comp_time);

		// if verification is needed
		if((argc ==4) && (0 == strcmp(argv[3], "-v"))){
			GET_TIME(t1);
        // get the serial output
			serial_mat_mul(A,B,serial_C);

			GET_TIME(t1);
			comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
			printf("N=%d: CPU Time(ms)=%.2f \n", MATRIX_DIM, comp_time);
 		// print_matrix(serial_C);
 		// print_matrix(C);

 		// Compare the reuslts
			compare_matrices(serial_C,C);

		}

	}
	else if (0 == strcmp(argv[1], "-c"))
	{

		long matrix_size=MATRIX_DIM*MATRIX_DIM*sizeof(Real);
 			// printf("%ld\n",matrix_size );

		GET_TIME(t1);
		
		Real* _A;
		gpuErrchk(cudaMalloc((void**) &_A, matrix_size));
 		// printStats();

		Real* _B;
		gpuErrchk(cudaMalloc((void**) &_B, matrix_size));
 		// printStats();

		Real* _C;
		gpuErrchk(cudaMalloc((void**) &_C, matrix_size));
 		// printStats();

 		// copy the matrices to device
		cudaMemcpy(_A, A, matrix_size, cudaMemcpyHostToDevice);
		cudaMemcpy(_B, B, matrix_size, cudaMemcpyHostToDevice);

 		// If the tiled mode needs to be enabled
		if (argc>2 && 0 == strcmp(argv[2], "-t")){
			printf("cuda tiled mode\n");

 			// set the grid and block sizes
			dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
			dim3 dimGrid;
			dimGrid.x = (MATRIX_DIM + dimBlock.x - 1)/dimBlock.x;
			dimGrid.y = (MATRIX_DIM + dimBlock.y - 1)/dimBlock.y;

			// GET_TIME(t1);
 			// execute the workload in the GPU
			cuda_tiled_mat_mul<<<dimGrid , dimBlock>>>(_A,_B,_C);

 			// Copy back the result
			cudaMemcpy(C,_C,matrix_size,cudaMemcpyDeviceToHost);

			GET_TIME(t2);
			comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
			printf("N=%d: CUDA Time(ms)=%.2f \n", MATRIX_DIM, comp_time);

			
			// if verification is needed
			if((argc ==4) && (0 == strcmp(argv[3], "-v"))){
				GET_TIME(t1);
 			// get the serial output
				serial_mat_mul(A,B,serial_C);

				GET_TIME(t2);
				comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
				printf("N=%d: CPU Time(ms)=%.2f \n", MATRIX_DIM, comp_time);

 				// print_matrix(serial_C);
 				// print_matrix(C);

 			// Compare the reuslts
				compare_matrices(serial_C,C);
			}

			

 			// free device memory
			cudaFree(_A);
			cudaFree(_B);
			cudaFree(_C);

		}
		else{
			printf("cuda mode\n");

			int K=100;			

			dim3 threadBlock(BLOCK_SIZE,BLOCK_SIZE);
			dim3 grid(K,K);

			// GET_TIME(t1);
 			// call the GPU
			cuda_simple_mat_mul<<<grid,threadBlock>>>(_A,_B,_C);

 			// Copy back the result
			cudaMemcpy(C,_C,matrix_size,cudaMemcpyDeviceToHost);

			GET_TIME(t2);
			comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
			printf("N=%d: CUDA Time(ms)=%.2f \n", MATRIX_DIM, comp_time);

			// if verification is needed
			if((argc ==3) && (0 == strcmp(argv[2], "-v"))){
				GET_TIME(t1);
 				// get the serial output
				serial_mat_mul(A,B,serial_C);

				GET_TIME(t2);
				comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
				printf("N=%d: CPU Time(ms)=%.2f \n", MATRIX_DIM, comp_time);

	 			// print_matrix(serial_C);
 				// print_matrix(C);

				compare_matrices(serial_C,C);

			}


			

 			// free device memory
			cudaFree(_A);
			cudaFree(_B);
			cudaFree(_C);


		}

	}
	else{
		print_usage();
	}
	return 0;
}
