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
// CUDA related
#define THREADS_PER_BLOCK 256
#define CALCS_PER_THREAD 50
// PThread related
#define MAX_PTHREADS 8
#define VECTOR_SIZE 10000000  //1e7
// #define VECTOR_SIZE 5  //1e8
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

void initialize_vector(Real vector[]){
  for(long i = 0; i < VECTOR_SIZE; i++){
    vector[i] = (rand() / (float) RAND_MAX) + 1;
  }
}

Real serial_calculation(Real vector1[], Real vector2[]){
  Real result = 0.0;
  for(long i = 0;i < VECTOR_SIZE; i++){
    result += vector1[i] * vector2[i];
  }
  return result;
}

__global__ void cuda_dot(Real *vector1, Real * vector2, Real *result){
  __shared__ Real temp[THREADS_PER_BLOCK];
  long index = threadIdx.x + blockDim.x * blockIdx.x;

  temp[threadIdx.x] = vector1[index] * vector2[index];

  __syncthreads

  if(0 == threadIdx.x){
    Real sum =0.0;
    for(long i = 0; i < THREADS_PER_BLOCK; i++){
      sum += temp[i];
    }
    atomicAdd(result,sum);
  }
}

__global__ void cuda_dot_product(Real *vector1, Real *vector2, Real *result) {
  unsigned long start_point = threadIdx.x + blockDim.x * blockIdx.x;
  // calculate the range to be multiplied
  long lowerbound = start_point * CALCS_PER_THREAD;
  long upperbound = lowerbound + CALCS_PER_THREAD - 1;
  // Don't try to calculate beyond vector size
  if(upperbound >= VECTOR_SIZE){
    upperbound = VECTOR_SIZE - 1;
  }
  __shared__ Real cache[THREADS_PER_BLOCK];
  // initialize the cache
  if(threadIdx.x == 0){
    for(int count = 0; count < THREADS_PER_BLOCK; count++){
      cache[count] = 0;
    }
  }
  Real sum = 0.0f;
  for(long index = lowerbound; index <= upperbound; index++){
    sum += vector1[index] * vector2[index];
  }
  // Wait till master has finished clearing the cache
  __syncthreads();
  // store the sum
  cache[threadIdx.x] = sum;
  sum = 0.0f;
  // should wait till everyone has finished computing
  __syncthreads();
  // take the sum of the elements
  if(threadIdx.x == 0){
    for(int count = 0; count < THREADS_PER_BLOCK; count++){
      sum += cache[count];
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
  // check the inputs and set the mode
  if(argc < 2){
    printf("Need Specify the Mode of Computation: Serial/Parallel/Cuda");
  }
  // initialize the vectors
  static Real vector1[VECTOR_SIZE];
  static Real vector2[VECTOR_SIZE];
  initialize_vector(vector1);
  initialize_vector(vector2);
  // if a serial execution is needed
  if(0 == strcmp(argv[1],"-s")){
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
  // if a parallel execution is needed
  else if(0 == strcmp(argv[1],"-p")){
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
    Real sum = 0.0;

    # pragma omp parallel for reduction(+:sum) num_threads(num_of_threads)
    for(int i = 0; i < VECTOR_SIZE; i++){
      sum += vector1[i] * vector2[i];
    }
    GET_TIME(t2);
    comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
    printf("Parallel Version N = %d: CPU Time(ms) = %.2f \n", VECTOR_SIZE, comp_time);
    #ifdef DP
      printf("Parallel result = %20.18f\n", sum);
    #else
      printf("Parallel result = %f\n", sum);
    #endif

    // if verification needed
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
  // if CUDA execution is needed
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
    cuda_dot<<<num_of_grids,THREADS_PER_BLOCK>>>(_vector1,_vector2,_results);
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
    // if verification needed
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
