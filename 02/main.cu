#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void random_ints(int *a, int n);

__global__ void add(int *a, int *b, int *c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

#define N 1
int main(void) {
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);

    printf("Starting...\n");

    // Allocate space for divide copies of a, b, c
    cudaMalloc((void **) &d_a, size);
    cudaMalloc((void **) &d_b, size);
    cudaMalloc((void **) &d_c, size);

    srand(time(NULL)); // get your seed!
    a = (int *) malloc(size);
    random_ints(a, N);
    b = (int *) malloc(size);
    random_ints(b, N);
    c = (int *) malloc(size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    add<<<N,1>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // printf("Result of c:\n");
    // for (int i = 0; i < N; i++) {
    //     printf("%d + %d = %d\n", a[i], b[i], c[i]);
    // }

    printf("%d + %d = %d\n", a[N-1], b[N-1], c[N-1]);

    printf("Done.\n");

    free(a); free(b); free(c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

	return 0;
}

// CPU function to generate a vector of random integers
void random_ints(int *a, int n) {
    for (int i = 0; i < n; i++)
        a[i] = rand() % 10000; // random number between 0 and 9999
}
