// Source: http://web.mit.edu/pocky/www/cudaworkshop/MonteCarlo/Pi.cu

// Written by Barry Wilkinson, UNC-Charlotte. Pi.cu  December 22, 2010.
//Derived somewhat from code developed by Patrick Rogers, UNC-C
//
//How to run?
//===========
//
//Single precision :
//
//nvcc -O3 pi-curand.cu ; ./a.out <thread_num>
//
//Double precision
//
//nvcc -O3 -D DP pi-curand.cu ; ./a.out <thread_num>

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <pthread.h>

#define TRIALS_PER_THREAD 4096
#define BLOCKS 256
#define THREADS 256


//Help code for switching between Single Precision and Double Precision
#ifdef DP
	typedef double Real;
	#define PI  3.14159265358979323846  // known value of pi
#else
	typedef float Real;
	#define PI 3.1415926535  // known value of pi
#endif


/**
A random number generator. 
Guidance from from http://stackoverflow.com/a/3067387/1281089
**/
Real randNumGen(){

   int random_value = rand(); //Generate a random number   
   Real unit_random = random_value / (Real) RAND_MAX; //make it between 0 and 1 
   return unit_random;
}

//struct for parameter passing between pthread calls
struct pthread_arg_struct {
    int tid;
    int total_threads;
    double total_tasks;
};


/**
The task allocated to a thread
**/
void *doCalcs(void *arguments)
{
	struct pthread_arg_struct *args = (struct pthread_arg_struct *)arguments;

	double total_threads = args -> total_threads;
	
	double total_tasks=args -> total_tasks; //total number of tasks
   int tid = args -> tid;       //obtain the value of thread id
   // printf("tid %d %lf\n", tid, total_tasks);

   //using malloc for the return variable in order make
   //sure that it is not destroyed once the thread call is finished
   double *in_count = (double *)malloc(sizeof(double));
   *in_count=0;
   
   //get the total number of iterations for a thread
   double tot_iterations= total_tasks/total_threads;
   // printf("%lf\n", tot_iterations);
   
   long counter=0;
   
   //calculation
   for(counter=0;counter<tot_iterations;counter++){
      Real x = randNumGen();
      Real y = randNumGen();
      
      Real result = sqrt((x*x) + (y*y));
      
      if(result<1){
         *in_count+=1;         //check if the generated value is inside a unit circle
      }
      
   }
   
   //get the remaining iterations calculated by thread 0
   if(tid==0){
      // int remainder = total_tasks%total_threads;
      double remainder = fmod((double)total_tasks,total_threads);
      // printf("%lf remainder\n", remainder );
      
      for(counter=0;counter<remainder;counter++){
      Real x = rand_r((unsigned int*) &tid) / (Real) RAND_MAX;
      Real y = rand_r((unsigned int*) &tid) / (Real) RAND_MAX;
      
      Real result = sqrt((x*x) + (y*y));
      
      if(result<1){
         *in_count+=1;         //check if the generated value is inside a unit circle
      }
      
   }
   }


   // printf("In count from #%d : %lf\n",tid,*in_count);
   pthread_exit((void *)in_count);     //return the in count
}



__global__ void gpu_monte_carlo(Real *estimate, curandState *states, int trials) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int points_in_circle = 0;
	Real x, y;

	curand_init(1234, tid, 0, &states[tid]);  // 	Initialize CURAND


	for(int i = 0; i < trials; i++) {
		x = curand_uniform (&states[tid]);
		y = curand_uniform (&states[tid]);
		points_in_circle += (x*x + y*y <= 1.0f); // count if x & y is in the circle.
	}
	estimate[tid] = 4.0f * points_in_circle / (Real) trials; // return estimate of pi
}

Real host_monte_carlo(long trials) {
	Real x, y;
	long points_in_circle;
	for(long i = 0; i < trials; i++) {
		x = rand() / (Real) RAND_MAX;
		y = rand() / (Real) RAND_MAX;
		points_in_circle += (x*x + y*y <= 1.0f);
	}
  // printf("Serial- points_in_circle : %ld\n", points_in_circle);
  // printf("Serial- trials: %ld\n",trials );
	return 4.0f * points_in_circle / trials;
}

int main (int argc, char *argv[]) {
	clock_t start, stop;

	//get the total number of pthreads
	int total_threads=atoi(argv[1]);

	pthread_t threads[total_threads];
   	int rc;
   	long t;
   	void *status;
   	double tot_in=0;
   	long total_tasks=pow(2,28);

   	int trials_per_thread= total_tasks/(BLOCKS*THREADS);

	Real host[BLOCKS * THREADS];
	Real *dev;
	curandState *devStates;

	printf("# of trials per thread = %d, # of blocks = %d, # of threads/block = %d.\n", trials_per_thread,
BLOCKS, THREADS);

	start = clock();

	cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(Real)); // allocate device mem. for counts
	
	cudaMalloc( (void **)&devStates, THREADS * BLOCKS * sizeof(curandState) );

	gpu_monte_carlo<<<BLOCKS, THREADS>>>(dev, devStates,trials_per_thread);

	cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(Real), cudaMemcpyDeviceToHost); // return results 

	Real pi_gpu;
	for(int i = 0; i < BLOCKS * THREADS; i++) {
		pi_gpu += host[i];
	}

	pi_gpu /= (BLOCKS * THREADS);

	stop = clock();

	// #ifdef DP
	// 	printf("GPU pi calculated in %20.18f s.\n", (stop-start)/(Real)CLOCKS_PER_SEC);

	// #else
		printf("GPU pi calculated in %f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);

	// #endif
	

	// PThreads
	start = clock();
	for(t=0;t<total_threads;t++){
		struct pthread_arg_struct* args=(struct pthread_arg_struct*)malloc(sizeof *args);
		args->total_threads=total_threads;
		args->tid=t;
		// args->total_tasks=BLOCKS*THREADS*TRIALS_PER_THREAD;
    args->total_tasks=total_tasks;
     	rc = pthread_create(&threads[t], NULL, doCalcs, (void *)args);
     	if (rc){
       		printf("ERROR; return code from pthread_create() is %d\n", rc);
       		exit(-1);
       	}
    }

  	//join the threads
   	for(t=0;t<total_threads;t++){
           
      pthread_join(threads[t], &status);
	    tot_in+=*(double*)status;            //keep track of the total in count 
      // printf("tot_in IN LOOP: %lf\n", tot_in);  
     }
   // printf("tot_in : %lf\n", tot_in);
   // printf("total_tasks : %ld\n", total_tasks);  

   Real pthread_pi=4*(tot_in/total_tasks);
   stop = clock();
 //   #ifdef DP
	// 	printf("PThreads pi calculated in %20.18f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);

	// #else
		printf("PThreads pi calculated in %f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);

	// #endif
   //End of PThreads 
	

	start = clock();
	Real pi_cpu = host_monte_carlo(total_tasks);
	stop = clock();

	// #ifdef DP
	// 	printf("CPU pi calculated in %20.18f s.\n", (stop-start)/(Real)CLOCKS_PER_SEC);

	// #else
		printf("CPU pi calculated in %f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);

	// #endif

	
	#ifdef DP
		printf("CUDA estimate of PI = %20.18f [error of %20.18f]\n", pi_gpu, pi_gpu - PI);
		printf("CPU estimate of PI = %20.18f [error of %20.18f]\n", pi_cpu, pi_cpu - PI);
		printf("PThread estimate of PI = %20.18f [error of %20.18f]\n",pthread_pi,pthread_pi - PI);

	#else
		printf("CUDA estimate of PI = %f [error of %f]\n", pi_gpu, pi_gpu - PI);
		printf("CPU estimate of PI = %f [error of %f]\n", pi_cpu, pi_cpu - PI);
		printf("PThread estimate of PI = %f [error of %f]\n",pthread_pi,pthread_pi - PI);

	#endif
	/* Last thing that main() should do */
   // pthread_exit(NULL);

   return 0;
}
