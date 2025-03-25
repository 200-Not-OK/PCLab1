/* 
This program will numerically compute the integral of
                  4/(1+x*x)                  
from 0 to 1. The value of this integral is pi. 
It uses OpenMP for parallel implementations.
*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static long num_steps = 1000000;

double compute_pi_serial(double step);
double compute_pi_false_sharing(double step);
double compute_pi_race_condition(double step);
double compute_pi_fixed(double step);
void usage(char prog_name[]);

int main(int argc, char **argv) {
    double start_time, run_time = 0, pi, step;
    int iter, method;

    if (argc != 3) {
        usage(argv[0]);
        exit(-1);
    }
    
    iter = atoi(argv[1]);
    method = atoi(argv[2]);
    step = 1.0 / (double)num_steps;

    for (int i = 0; i < iter; i++) {
        start_time = omp_get_wtime();
        
        if (method == 1) {
            pi = compute_pi_false_sharing(step);
        } else if (method == 2) {
            pi = compute_pi_race_condition(step);
        } else if (method == 3) {
            pi = compute_pi_fixed(step);
        } else {
            pi = compute_pi_serial(step);
        }
        
        run_time += omp_get_wtime() - start_time;
    }
    
    printf("\npi with %ld steps is %f in %f seconds\n", num_steps, pi, run_time / iter);
    return EXIT_SUCCESS;
}

// Sequential computation of pi
double compute_pi_serial(double step) {
    double pi, x, sum = 0.0;
    for (int i = 1; i <= num_steps; i++) {
        x = (i - 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    return pi;
}

// Parallel computation with false sharing
double compute_pi_false_sharing(double step) {
    int i, id, nthreads;
    double x, pi;
    int num_threads = omp_get_max_threads();
    double *sum = (double *)malloc(num_threads * sizeof(double));
    
    if (!sum) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    
    omp_set_num_threads(num_threads);
    #pragma omp parallel private(id, x, i)
    {
        id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        if (id == 0) nthreads = num_threads;

        sum[id] = 0.0;
        for (i = id; i < num_steps; i += num_threads) {
            x = (i + 0.5) * step;
            sum[id] += 4.0 / (1.0 + x * x);
        }
    }

    pi = 0.0;
    for (i = 0; i < nthreads; i++)
        pi += step * sum[i];
    
    free(sum);
    return pi;
}

// Parallel computation with race condition
double compute_pi_race_condition(double step) {
    int i;
    double x, pi = 0.0, sum = 0.0;
    
    #pragma omp parallel for private(x) shared(sum)
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x); // Race condition occurs here
    }
    
    pi = step * sum;
    return pi;
}

// Parallel computation with race condition eliminated
double compute_pi_fixed(double step) {
    int i;
    double x, pi = 0.0, sum = 0.0;
    int num_threads = omp_get_max_threads();
    
    #pragma omp parallel private(x, i)
    {
        double local_sum = 0.0;
        int id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        
        for (i = id; i < num_steps; i += num_threads) {
            x = (i + 0.5) * step;
            local_sum += 4.0 / (1.0 + x * x);
        }
        
        #pragma omp critical
        sum += local_sum;
    }
    
    pi = step * sum;
    return pi;
}

// Usage information
void usage(char prog_name[]) {
    fprintf(stderr, "usage: %s <number of times to run> <method>\n", prog_name);
    fprintf(stderr, "Method options:\n");
    fprintf(stderr, "0 - Serial\n");
    fprintf(stderr, "1 - False Sharing\n");
    fprintf(stderr, "2 - Race Condition\n");
    fprintf(stderr, "3 - Fixed Race Condition\n");
}
