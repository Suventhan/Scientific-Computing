#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <curand_kernel.h>
#include <iostream>
#include <iomanip>

#define BLOCKS 256
#define THREADS 256
// #define TOTAL_TASKS pow(2,28)

//Help code for switching between Single Precision and Double Precision
#ifdef DP
typedef double Real;
  #define PI  3.14159265358979323846  // known value of pi
#else
typedef float Real;
  #define PI 3.1415926535  // known value of pi
#endif

// we could vary M & N to find the perf sweet spot

struct estimate_pi :
public thrust::unary_function<unsigned int, Real>
{
  const int trials;
  estimate_pi(int _trials) : trials(_trials){}
  __device__
  Real operator()(unsigned int thread_id)
  {
    Real sum = 0;
    // long total_tasks=pow(2,28);
    int N= trials;
    // unsigned int N = 10000; // samples per thread
    unsigned int seed = thread_id;
    curandState s;
    // seed a random number generator
    curand_init(seed, 0, 0, &s);

    // take N samples in a quarter circle
    for(unsigned int i = 0; i < N; ++i)
    {
      // draw a sample from the unit square
      Real x = curand_uniform(&s);
      Real y = curand_uniform(&s);

      // measure distance from the origin
      Real dist = sqrtf(x*x + y*y);

      // add 1.0f if (u0,u1) is inside the quarter circle
      if(dist <= 1.0f)
        sum += 1.0f;
    }

    // multiply by 4 to get the area of the whole circle
    sum *= 4.0f;

    // divide by N
    return sum / N;
  }
};

int main(void)
{
  clock_t start, stop;

  int M = BLOCKS*THREADS;

  long total_tasks=pow(2,28);
  int trials_per_thread= total_tasks/M;

  std::cout << "# of trials per thread = "<< trials_per_thread <<" # of blocks * # of threads/block = "
  << BLOCKS*THREADS << std::endl;

  start = clock();

  Real estimate = thrust::transform_reduce(
    thrust::counting_iterator<int>(0),
    thrust::counting_iterator<int>(M),
    estimate_pi(trials_per_thread),
    0.0f,
    thrust::plus<Real>());
  estimate /= M;

  stop = clock();

  #ifdef DP
  std::cout << std::setprecision(20);
  #else
  std::cout << std::setprecision(7);
  #endif

  std::cout << "THRUST pi calculated in " << (stop-start)/(float)CLOCKS_PER_SEC << " s."<< std::endl;
  std::cout << "THRUST estimate of PI = " << estimate << " [error of " << estimate - PI << "]" << std::endl;

  return 0;
}
