#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <omp.h>

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
	long points_in_circle = 0;
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
	long total_tasks=pow(2,28);

	int trials_per_thread= total_tasks/(BLOCKS*THREADS);

	Real host[BLOCKS * THREADS];
	Real *dev;
	curandState *devStates;

	printf("# of trials per thread = %d, # of blocks = %d, # of threads/block = %d.\n", trials_per_thread, BLOCKS, THREADS);

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
	printf("GPU pi calculated in %f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);
	start = clock();
	Real x,y;								//loop counter
  long points_in_circle = 0;							//Count holds all the number of how many good coordinates
	Real omp_pi = 0.f;							//holds approx value of pi

	#pragma omp parallel firstprivate(x, y, i) reduction(+:points_in_circle) num_threads(total_threads)
	{
		for (long i = 0; i < total_tasks; ++i)					//main loop
		{
			x = rand() / (Real) RAND_MAX;
			y = rand() / (Real) RAND_MAX;
			points_in_circle += (x*x + y*y <= 1.0f);
		}
	}
	omp_pi = 4.0f * points_in_circle / total_tasks;
	stop = clock();
	printf("OpenMP pi calculated in %f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);

	start = clock();
	Real pi_cpu = host_monte_carlo(total_tasks);
	stop = clock();

	printf("CPU pi calculated in %f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);
	#ifdef DP
	printf("CUDA estimate of PI = %20.18f [error of %20.18f]\n", pi_gpu, pi_gpu - PI);
	printf("CPU estimate of PI = %20.18f [error of %20.18f]\n", pi_cpu, pi_cpu - PI);
	printf("OpenMP estimate of PI = %20.18f [error of %20.18f]\n",omp_pi,omp_pi - PI);
	#else
	printf("CUDA estimate of PI = %f [error of %f]\n", pi_gpu, pi_gpu - PI);
	printf("CPU estimate of PI = %f [error of %f]\n", pi_cpu, pi_cpu - PI);
	printf("OpenMP estimate of PI = %f [error of %f]\n",omp_pi,omp_pi - PI);
	#endif
	return 0;
}
