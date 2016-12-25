# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>

int main ( int argc, char *argv[] );
double test01 ( int n, double x[], double y[] );
double test02 ( int n, double x[], double y[] );

int main ( int argc, char *argv[] )
{
  double factor;
  int i;
  int n;
  double wtime;
  double *x;
  double xdoty;
  double *y;

  printf ( "\n" );
  printf ( "DOT_PRODUCT\n" );
  printf ( "  C/OpenMP version\n" );
  printf ( "\n" );
  printf ( "  A program which computes a vector dot product.\n" );

  printf ( "\n" );
  printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
  printf ( "  Number of threads =              %d\n", omp_get_max_threads ( ) );
  n = 100;

  while ( n < 1000000 )
  {
    n = n * 10;

    x = ( double * ) malloc ( n * sizeof ( double ) );
    y = ( double * ) malloc ( n * sizeof ( double ) );

    factor = ( double ) ( n );
    factor = 1.0 / sqrt ( 2.0 * factor * factor + 3 * factor + 1.0 );

    for ( i = 0; i < n; i++ )
    {
      x[i] = ( i + 1 ) * factor;
    }

    for ( i = 0; i < n; i++ )
    {
      y[i] = ( i + 1 ) * 6 * factor;
    }

    printf ( "\n" );
    wtime = omp_get_wtime ( );

    xdoty = test01 ( n, x, y );

    wtime = omp_get_wtime ( ) - wtime;

    printf ( "  Sequential  %8d  %14.6e  %15.10f\n", n, xdoty, wtime );
    wtime = omp_get_wtime ( );

    xdoty = test02 ( n, x, y );

    wtime = omp_get_wtime ( ) - wtime;

    printf ( "  Parallel    %8d  %14.6e  %15.10f\n", n, xdoty, wtime );

    free ( x );
    free ( y );
  }
  printf ( "\n" );
  printf ( "DOT_PRODUCT\n" );
  printf ( "  Normal end of execution.\n" );

  return 0;
}
/******************************************************************************/

double test01 ( int n, double x[], double y[] )
{
  int i;
  double xdoty;

  xdoty = 0.0;

  for ( i = 0; i < n; i++ )
  {
    xdoty = xdoty + x[i] * y[i];
  }

  return xdoty;
}

double test02 ( int n, double x[], double y[] )
{
  int i;
  double xdoty;

  xdoty = 0.0;

# pragma omp parallel \
  shared ( n, x, y ) \
  private ( i )

# pragma omp for reduction ( + : xdoty )

  for ( i = 0; i < n; i++ )
  {
    xdoty = xdoty + x[i] * y[i];
  }

  return xdoty;
}
