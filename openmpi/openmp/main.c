#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>


struct timespec start_time;

void timer_start(void) {
    clock_gettime(CLOCK_REALTIME, &start_time);
}

double timer_end(void) {
    struct timespec end_time;

    clock_gettime(CLOCK_REALTIME, &end_time);

    return difftime(end_time.tv_sec, start_time.tv_sec) +
        (end_time.tv_nsec - start_time.tv_nsec) / 1000000000.;
}

void timer_report(void) {
    double timing = timer_end();
    printf("took %10.3f sec\n", timing);
}

void multimult( double *a, double *b, int len, int steps ) {
    double c;
    int t, i;

    for (t = 1; t <= steps; t++) {
        #pragma omp parallel for private( i) shared(a, b, len, steps) schedule(static)
        for (i = 0; i < len; i++) {
            c = a[i] * b[i];
            a[i] = c * (double) t;
        }
    }
}

void multimult2( double *a, double *b, int len, int steps ) {
    double c;

        int i;
    #pragma omp parallel shared(a, b, len, steps) private(i)
    for (int t = 1; t <= steps; t++) {
        #pragma omp for
        for (i = 0; i < len; i++) {
            c = a[i] * b[i];
            a[i] = c * (double) t;
        }
    }
}

void multimult3( double *a, double *b, int len, int steps ) {
    double c;
    int t, i;

    #pragma omp parallel for private( i) shared(a, b, len, steps) schedule(static)
    for (i = 0; i < len; i++) {
        for (t = 1; t <= steps; t++) {
            c = a[i] * b[i];
            a[i] = c * (double) t;
        }
    }
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
        printf("Usage: %s [vec_len] [timesteps] [method]", argv[0]);
    }

    srand(1);
    int vec_len = atoi(argv[1]);
    int timesteps = atoi(argv[2]);
    int method = 3;
    int threads = atoi(argv[3]);

    omp_set_num_threads(threads);


    double *a = (double*)malloc(vec_len*sizeof(double));
    double *b = (double*)malloc(vec_len*sizeof(double));

    for (int i = 0; i < vec_len; i++) {
        a[i] = 1 + (rand() / (double) RAND_MAX);
        b[i] = 1 + (rand() / (double)RAND_MAX);
    }

    if (method == 1) {
        timer_start();
        multimult(a, b, vec_len, timesteps);
        timer_report();
    }

    if (method == 2) {
        timer_start();
        multimult2(a, b, vec_len, timesteps);
        timer_report();
    }

    if (method == 3) {
        timer_start();
        multimult3(a, b, vec_len, timesteps);
        timer_report();
    }
    return EXIT_SUCCESS;
}

