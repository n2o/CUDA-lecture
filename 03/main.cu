/************************************************************************************
 * Matrix * Matrix Multiplication
 *
 * The GPU is with an input of n = 512 faster than the GPU
 ************************************************************************************/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void random_ints(float *a, int n);
void print_matrix(float *matrix, int n);
void matrix_mul_cpu(float *matA, float *matB, float *matC, int n);
void matrix_mul_gpu(float *matA, float *matB, float *matC, int n);
__global__ void matrix_mul_gpu_element(float *gpuA, float *gpuB, float *gpuC, int n);

#define N 1024
int main(int argc, char* argv[]) {
	if (argc < 2) {
		printf("Please provide the matrix size parameter n.\n");
		return 0;
	}
    srand(time(NULL)); // get your seed!

    // Get n, m from command line
    int n = atoi(argv[1]);

    // Fill matrix with random numbers
    float *matA = (float *) malloc(n * n * sizeof(float));
    random_ints(matA, n*n);

    float *matB = (float *) malloc(n * n * sizeof(float));
    random_ints(matB, n*n);

    // Store results in matrix C
    float *matC = (float *) malloc(n * n * sizeof(float));

	// It's CPU time!
//	printf("\n### CPU time! ###\n");
//	clock_t cputime = clock();
//	matrix_mul_cpu(matA, matB, matC, n);
//	printf("Matrix C [0]: \t\t%f\n", matC[0]);
//	printf("Matrix C [%d]: \t%f\n", n-1, matC[n-1]);
//	printf("Time: \t\t\t%f s\n", ((double)clock() - cputime) / CLOCKS_PER_SEC);
//
//	printf("\nResetting Matrix C... ");
//	random_ints(matC, n*n);
//	printf("Done.\n");

    // GPU
    printf("\n### Now the GPU... ###\n");
    clock_t gputime = clock();
    matrix_mul_gpu(matA, matB, matC, n);
    printf("Matrix C [0]: \t\t%f\n", matC[0]);
    printf("Matrix C [%d]: \t%f\n", n-1, matC[n-1]);
    printf("Time: \t\t\t%f s\n", ((double)clock() - gputime) / CLOCKS_PER_SEC);

    free(matA);
    free(matB);
    free(matC);

	return 0;
}

// GPU Version
void matrix_mul_gpu(float *matA, float *matB, float *matC, int n) {
	unsigned int threads = 32;
	unsigned int blocks = (n + (threads - 1)) / threads;
	dim3 BlocksPerGrid(blocks, blocks);
	dim3 ThreadsPerBlock(threads, threads);

	int size = n * n * sizeof(float);
	float *gpuA, *gpuB, *gpuC;

	// Allocate and load matrix A and B to the gpu
	cudaMalloc(&gpuA, size);
	cudaMemcpy(gpuA, matA, size, cudaMemcpyHostToDevice);
	cudaMalloc(&gpuB, size);
	cudaMemcpy(gpuB, matB, size, cudaMemcpyHostToDevice);

	cudaMalloc(&gpuC, size);

	// Launch the device in one block with n Threads
	// matrix_mul_gpu_element<<<dim3(1,1,1), dim3(n,1,1)>>>(gpuA, gpuB, gpuC, n);
	matrix_mul_gpu_element<<<BlocksPerGrid, ThreadsPerBlock>>>(gpuA, gpuB, gpuC, n);

	// Get result from device
	cudaMemcpy(matC, gpuC, size, cudaMemcpyDeviceToHost);

	cudaFree(gpuA);
	cudaFree(gpuB);
	cudaFree(gpuC);
}

__global__ void matrix_mul_gpu_element(float *gpuA, float *gpuB, float *gpuC, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float a, b, sum = 0;
	for (int k = 0; k < n; ++k) {
		a = gpuA[k + row*n];
		b = gpuB[col + k*n];
		sum += a * b;
	}
	gpuC[col + row*n] = sum;
}

// CPU Version
void matrix_mul_cpu(float *matA, float *matB, float *matC, int n) {
	float a, b, sum;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			sum = 0;
			for (int k = 0; k < n; ++k) {
				a = matA[k + i*n];
				b = matB[j + k*n];
				sum += a * b;
			}
			matC[j + i*n] = sum;
		}
	}
}

//////////////////////////////////
/* Functions for initialization */
//////////////////////////////////

// CPU function to generate a vector of random integers
void random_ints(float* a, int n) {
    for (int i = 0; i < n; i++)
        a[i] = rand() % 10000; // random number between 0 and 9999
}

void print_matrix(float* matrix, int n) {
	for (int i = 0; i < n*n; i++) {
		if (i % n == 0) printf("\n");
		printf("%f ", matrix[i]);
	}
}
