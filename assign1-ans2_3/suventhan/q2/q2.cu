#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <cuda_runtime_api.h>
#include <errno.h>
#include <omp.h>

#define GET_TIME(x);  if (clock_gettime(CLOCK_MONOTONIC, &(x)) < 0) \
{ perror("clock_gettime( ):"); exit(EXIT_FAILURE); }

// For CUDA
#define THREADS_PER_BLOCK 256
#define CALCS_PER_THREAD 50
// For OpenMP
#define MAX_PTHREADS 8
#define VECTOR_SIZE 10000000
// #define VECTOR_SIZE 50000000
// #define VECTOR_SIZE 100000000

//Switching between Single Precision and Double Precision
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

// Vector initialization using random numbers between 1 and 2
void initialize_vector(Real vector[]){
  for(long i = 0; i < VECTOR_SIZE; i++){
    vector[i] = (rand() / (float) RAND_MAX) + 1;
  }
}

//Sequential Calculation of dot product
Real sequential_calculation(Real vector1[], Real vector2[]){
  Real result = 0.0;
  for(long i = 0;i < VECTOR_SIZE; i++){
    result += vector1[i] * vector2[i];
  }
  return result;
}

// GPU parallel computation of dot product
__global__ void cuda_dot_product(Real *vector1, Real *vector2, Real *result) {
  
  unsigned long initial = threadIdx.x + blockDim.x * blockIdx.x;
  // calculate the range to be multiplied
  long start = initial * CALCS_PER_THREAD;
  long end = start + CALCS_PER_THREAD - 1;
  // ensure to calculate within vector size
  if(end >= VECTOR_SIZE){
    end = VECTOR_SIZE - 1;
  }
  __shared__ Real temp[THREADS_PER_BLOCK];
  // initialize the temp to 0
  if(threadIdx.x == 0){
    for(int count = 0; count < THREADS_PER_BLOCK; count++){
      temp[count] = 0;
    }
  }
  Real sum = 0.0f;
  for(long index = start; index <= end; index++){
    sum += vector1[index] * vector2[index];
  }
  // Wait until all threads to finished
  __syncthreads();
  // store the sum to temp
  temp[threadIdx.x] = sum;
  sum = 0.0f;
  // Wait until all threads to finished
  __syncthreads();
  // take the sum of the elements
  if(threadIdx.x == 0){
    for(int count = 0; count < THREADS_PER_BLOCK; count++){
      sum += temp[count];
    }
    result[blockIdx.x] = sum;
  }
}

int main(int argc, char const *argv[])
{
  struct timespec t1, t2;
  long sec, nsec;
  float comp_time;        // in milli seconds
  // Initialize the random seed
  srand(time(NULL));
  // check the command line arguements to select the computation mode
  if(argc < 2){
    printf("Need Specify the Mode of Computation: Sequential/Parallel/Cuda");
  }
  // define and initialize the vectors
  static Real vector1[VECTOR_SIZE];
  static Real vector2[VECTOR_SIZE];
  initialize_vector(vector1);
  initialize_vector(vector2);
  // check to run the sequential computation
  if(0 == strcmp(argv[1],"-s")){
    GET_TIME(t1);
    Real result = sequential_calculation(vector1,vector2);
    GET_TIME(t2);
    comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
    printf("Sequential Version \n N = %d: CPU Time(ms) = %.2f \n", VECTOR_SIZE, comp_time);
    #ifdef DP
      printf("Sequential result = %20.18f\n", result);
    #else
      printf("Sequential result = %f\n", result);
    #endif
  }
  // check to run the cpu parallel computation
  else if(0 == strcmp(argv[1],"-p")){
    int num_of_threads;
    // check whether the # of threads given as runtime arguements
    if(argc < 3){
      printf("Need to specify the # of threads at runtime");
      return -1;
    }
    num_of_threads = atoi(argv[2]);
    if(num_of_threads > MAX_PTHREADS){
      printf("Up to 8 threads can be created\n");
      return -1;
    }

    GET_TIME(t1);
    Real sum = 0.0;
    //cpu parallel version code using OpenMP
    # pragma omp parallel for reduction(+:sum) num_threads(num_of_threads)
    for(int i = 0; i < VECTOR_SIZE; i++){
      sum += vector1[i] * vector2[i];
    }
    GET_TIME(t2);
    comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
    printf("CPU Parallel Version N = %d: CPU Time(ms) = %.2f \n", VECTOR_SIZE, comp_time);
    #ifdef DP
      printf("CPU Parallel result = %20.18f\n", sum);
    #else
      printf("CPU Parallel result = %f\n", sum);
    #endif

    // For verification against sequential version
    if(argc == 4 && 0 == strcmp(argv[3],"-v")){
      GET_TIME(t1);
      Real result = serial_calculation(vector1,vector2);
      GET_TIME(t2);
      comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
      printf("Serial Version \n N = %d: CPU Time(ms) = %.2f \n", VECTOR_SIZE, comp_time);
      #ifdef DP
        printf("Serial result = %20.18f\n", result);
      #else
        printf("Serial result = %f\n", result);
      #endif
    }
  }
  // check to run the CUDA computation
  else if(0 == strcmp(argv[1],"-c")){
    //Allocate vectors in device memory
    size_t size = VECTOR_SIZE * sizeof(Real);
    Real* _vector1;
    GET_TIME(t1);
    cudaMalloc((void**) &_vector1, size);
    Real* _vector2;
    cudaMalloc((void**) &_vector2, size);
    //copy vectors from host memory to device memory
    cudaMemcpy(_vector1, vector1,size,cudaMemcpyHostToDevice);
    cudaMemcpy(_vector2, vector2,size,cudaMemcpyHostToDevice);
    long num_of_grids = (VECTOR_SIZE/(THREADS_PER_BLOCK*CALCS_PER_THREAD))+1;
    // Allocate memory for results in the host memory
    Real results[num_of_grids];
    Real* _results;
    size_t result_size = sizeof(Real)*num_of_grids;
    cudaMalloc((void**) &_results, result_size);
    // carry out the calculations
    cuda_dot_product<<<num_of_grids,THREADS_PER_BLOCK>>>(_vector1,_vector2,_results);
    // copy the results back from the device memory to host memory
    cudaMemcpy(results,_results, sizeof(Real)*num_of_grids,cudaMemcpyDeviceToHost);
    GET_TIME(t2);
    comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
    printf("CUDA Version N = %d: CPU Time(ms) = %.2f \n", VECTOR_SIZE, comp_time);
    // free device memory
    cudaFree(_vector1);
    cudaFree(_vector2);
    cudaFree(_results);
    // calculate the final result
    Real result = 0;
    for(long i= 0; i < num_of_grids; i++){
      result += results[i];
    }
    #ifdef DP
      printf("CUDA result = %20.18f\n", result);
    #else
      printf("CUDA result = %f\n", result);
    #endif
    // For verification against sequential
    if(argc == 3 && 0 == strcmp(argv[2],"-v")){
      GET_TIME(t1);
      Real result = serial_calculation(vector1,vector2);
      GET_TIME(t2);
      comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
      printf("Serial Version \n N = %d: CPU Time(ms) = %.2f \n", VECTOR_SIZE, comp_time);
      #ifdef DP
        printf("Serial result = %20.18f\n", result);
      #else
        printf("Serial result = %f\n", result);
      #endif
    }
  }
  else{
    printf("Need Specify the Mode of Computation: Serial/Parallel/Cuda");
  }
  return 0;
}
