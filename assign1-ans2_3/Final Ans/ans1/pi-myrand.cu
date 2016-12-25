#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>

// for 2^24
// #define TRIALS_PER_THREAD 256
// for 2^26
// #define TRIALS_PER_THREAD 1024
// for 2^28
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

__device__ Real my_rand(unsigned int *seed) {
	unsigned long a = 16807;  // constants for random number generator
	unsigned long m = 2147483647;   // 2^31 - 1
	unsigned long x = (unsigned long) *seed;

	x = (a * x)%m;
	*seed = (unsigned int) x;
	return ((Real)x)/m;
}

__global__ void gpu_monte_carlo(Real *estimate) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int points_in_circle = 0;
	Real x, y;

	unsigned int seed =  tid + 1;  // starting number in random sequence

	for(int i = 0; i < TRIALS_PER_THREAD; i++) {
		x = my_rand(&seed);
		y = my_rand(&seed);
		points_in_circle += (x*x + y*y <= 1.0f); // count if x & y is in the circle.
	}
	estimate[tid] = 4.0f * points_in_circle / (Real) TRIALS_PER_THREAD; // return estimate of pi
}

Real host_monte_carlo(long trials) {
	Real x, y;
	long points_in_circle;
	for(long i = 0; i < trials; i++) {
		x = rand() / (Real) RAND_MAX;
		y = rand() / (Real) RAND_MAX;
		points_in_circle += (x*x + y*y <= 1.0f);
	}
	return 4.0f * points_in_circle / trials;
}

int main (int argc, char *argv[]) {
	clock_t start, stop;
	Real host[BLOCKS * THREADS];
	Real *dev;

	printf("# of trials per thread = %d, # of blocks = %d, # of threads/block = %d.\n", TRIALS_PER_THREAD, BLOCKS, THREADS);

	start = clock();

	cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(Real)); // allocate device mem. for counts
	gpu_monte_carlo<<<BLOCKS, THREADS>>>(dev);
	cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(Real), cudaMemcpyDeviceToHost); // return results

	Real pi_gpu;
	for(int i = 0; i < BLOCKS * THREADS; i++) {
		pi_gpu += host[i];
	}
	pi_gpu /= (BLOCKS * THREADS);

	stop = clock();

	printf("GPU pi calculated in %f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);

	start = clock();
	Real pi_cpu = host_monte_carlo(BLOCKS * THREADS * TRIALS_PER_THREAD);
	stop = clock();
	printf("CPU pi calculated in %f s.\n", (stop-start)/(Real)CLOCKS_PER_SEC);

	#ifdef DP
	printf("CUDA estimate of PI = %20.18f [error of %20.18f]\n", pi_gpu, pi_gpu - PI);
	printf("CPU estimate of PI = %20.18f [error of %20.18f]\n", pi_cpu, pi_cpu - PI);
	#else
	printf("CUDA estimate of PI = %f [error of %f]\n", pi_gpu, pi_gpu - PI);
	printf("CPU estimate of PI = %f [error of %f]\n", pi_cpu, pi_cpu - PI);
	#endif

	return 0;
}
