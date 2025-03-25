/* 
This program will numerically compute the integral of
                  4/(1+x*x)				  
from 0 to 1.  The value of this integral is pi. 
It uses the timer from the OpenMP runtime library
*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// #define NUM_THREADS 12

double compute_pi(double step);
double compute_pi_1(double step);
double compute_pi_2(double step);
double compute_pi_3(double step);
void usage(char prog_name[]);
static long num_steps = 1000000;

int main (int argc, char **argv)
{
	double start_time, run_time=0, pi, step;
	int iter;

	if (argc != 2) {
		usage(argv[0]);
		exit (-1);
	}
	
	iter=atoi(argv[1]);
	step = 1.0/(double)num_steps;
	int num_threads = omp_get_max_threads();
	for(int i=0; i<iter; i++){
        /* Record the time for staring the computation of pi */
		start_time = omp_get_wtime();
		pi=compute_pi(step);
        /* Record the time for ending of the computation of pi and
		find the elapsed time for the computation of pi
		Note we are running the same process for multiple time and taking the average here */
		run_time += omp_get_wtime() - start_time;
	}
	double timeSerial = run_time/iter;
	
	printf("\nSequential: pi with %ld steps is %f in %f seconds using %d threads\n",num_steps,pi,timeSerial,num_threads);

	//METHOD 1: Parallel

	run_time=0;
	for(int i=0; i<iter; i++){
        /* Record the time for staring the computation of pi */
		start_time = omp_get_wtime();
		pi=compute_pi_1(step);
        /* Record the time for ending of the computation of pi and
		find the elapsed time for the computation of pi
		Note we are running the same process for multiple time and taking the average here */
		run_time += omp_get_wtime() - start_time;
	}
	double timeParallel1 = run_time/iter;
	double speedup1 = timeSerial/timeParallel1;
	printf("\nParallel with false sharing: pi with %ld steps is %f in %f seconds using %d threads\n",num_steps,pi,run_time/iter,num_threads);
	printf("Speedup: %f\n", speedup1);

	//METHOD 2: Parallel

	run_time=0;
	for(int i=0; i<iter; i++){
        /* Record the time for staring the computation of pi */
		start_time = omp_get_wtime();
		pi=compute_pi_2(step);
        /* Record the time for ending of the computation of pi and
		find the elapsed time for the computation of pi
		Note we are running the same process for multiple time and taking the average here */
		run_time += omp_get_wtime() - start_time;
	}
	double timeParallel2 = run_time/iter;
	double speedup2 = timeSerial/timeParallel2;
	printf("\nParallel with race condition: pi with %ld steps is %f in %f seconds using %d threads\n",num_steps,pi,run_time/iter,num_threads);
	printf("Speedup: %f\n", speedup2);

	//METHOD 3: Parallel

	run_time=0;
	for(int i=0; i<iter; i++){
        /* Record the time for staring the computation of pi */
		start_time = omp_get_wtime();
		pi=compute_pi_3(step);
        /* Record the time for ending of the computation of pi and
		find the elapsed time for the computation of pi
		Note we are running the same process for multiple time and taking the average here */
		run_time += omp_get_wtime() - start_time;
	}
	double timeParallel3 = run_time/iter;
	double speedup3 = timeSerial/timeParallel3;
	printf("\nParallel with no race condition: pi with %ld steps is %f in %f seconds using %d threads\n",num_steps,pi,run_time/iter,num_threads);
	printf("Speedup: %f\n", speedup3);
    return EXIT_SUCCESS; 
}

/*--------------------------------------------------------------------
 * Function:    compute_pi
 * Purpose:     Compute number pi in serial
 * In arg:      step
 */  
double compute_pi(double step){
	double pi, x, sum=0.0;
	for (int i=1;i<= num_steps; i++){
		x = (i - 0.5)*step;
		sum = sum + 4.0/(1.0+x*x);
	}
	pi = step * sum;
	return pi;
} /* compute_pi */

double compute_pi_1(double step){
	int nthreads;
	double pi = 0.0;
	int num_threads = omp_get_max_threads();
    double *sum = (double *)malloc(num_threads * sizeof(double));
	omp_set_num_threads(num_threads);
	#pragma omp parallel
	{
		int i, id, total_threads;
		double x;
		total_threads = omp_get_num_threads();
		id = omp_get_thread_num();
		if(id == 0) nthreads = total_threads;
		for (i=id, sum[id]=0.0; i < num_steps; i += total_threads){
			x = (i + 0.5)*step;
			sum[id] += 4.0/(1.0+x*x);
		}
	}
	for (int i = 0; i < nthreads; i++)
	{
		pi += step * sum[i];
	}
	return pi;
}

double compute_pi_2(double step){
	int num_threads = omp_get_max_threads();
	omp_set_num_threads(num_threads);
	int total_threads, id;
	double pi = 0.0, x, sum = 0.0;
	#pragma omp parallel shared(sum)
	{
		total_threads = omp_get_num_threads();
		id = omp_get_thread_num();

		for (int i=id; i <= num_steps; i += total_threads){
			x = (i + 0.5)*step;
			sum += 4.0/(1.0+x*x);
		}
	}
	pi = step * sum;
	return pi;
}

double compute_pi_3(double step){
	int num_threads = omp_get_max_threads();
	omp_set_num_threads(num_threads);
	int i;
	double pi = 0.0, x, sum = 0.0;

	#pragma omp parallel private(x, i)
	{
		double local_sum = 0.0;
		int total_threads = omp_get_num_threads();
		int id = omp_get_thread_num();
		
		for (i=id; i < num_steps; i += total_threads){
			x = (i + 0.5)*step;
			local_sum += 4.0/(1.0 + x * x);
		}
		#pragma omp critical
		sum += local_sum;
	}
	pi = step * sum;
	return pi;
}


/*--------------------------------------------------------------------
 * Function:    usage
 * Purpose:     Print command line for function
 * In arg:      prog_name
 */
void usage(char prog_name[]) {
   fprintf(stderr, "usage:  %s <number of times to run>\n", prog_name);
} /* usage */


