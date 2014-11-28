
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void vecSum_GPU3(const double *a, double *res, const unsigned long n);
__global__ void vecSum_betterGPU(const double *a, double *gpuRes, const unsigned long n, int offset);

void random_double(double *a, int n);

int main(int argc, char* argv[]) {
    double n = atof(argv[1]);
    printf("n = %f\n", n);

    cudaDeviceProp devProperties;
    cudaGetDeviceProperties(&devProperties, 0);
    unsigned int ThreadsPerBlock = devProperties.maxThreadsPerBlock;
    unsigned int BlocksPerGrid = (n + (devProperties.maxThreadsPerBlock-1)) / devProperties.maxThreadsPerBlock;

    int size = n * sizeof(double);
    int resSize = n * sizeof(double) / 1024;
    double *vec = (double *) malloc(size);

    double *pinnedVec;
    cudaMallocHost(&pinnedVec, size);

    double *gpuVec;
    double *gpuRes;                             // gpuRes type is (maybe) not correct
    // double cpuRes = 0;
    double *res = (double *) malloc(resSize);

    // srand(time(NULL)); // get your seed!
    // random_double(vec, n);
    // memcpy(pinnedVec, vec, size);

    for (int i = 0; i < (int) n; i++) {
        pinnedVec[i] = 1;
    }
    // CPU Time
    // clock_t cputime = clock();
    // cpuRes = 0;
    // for (int i = 0; i < n; i++) {
    //  cpuRes += vec[i];
    // }
    // printf("CPU Result: %f\n", cpuRes);
    // printf("Time: %f s\n", ((double)clock() - cputime) / CLOCKS_PER_SEC);

    // GPU Time
    clock_t gputime = clock();

    cudaMalloc(&gpuVec, size);
    cudaMalloc(&gpuRes, size);
    cudaMemcpy(gpuVec, pinnedVec, size, cudaMemcpyHostToDevice);

    int numOfRuns = ceil(n / (1024*1024));
    printf("numOfRuns = %d\n", numOfRuns);

    for (int i = 0; i < numOfRuns; i++) {
        vecSum_betterGPU<<<BlocksPerGrid, ThreadsPerBlock>>>((double *)(gpuVec + i*(1024*1024)), gpuRes, n, i);
        cudaMemcpy(gpuVec, gpuRes, size, cudaMemcpyDeviceToDevice);
        vecSum_betterGPU<<<BlocksPerGrid, ThreadsPerBlock>>>(gpuVec, gpuRes, n, i);
    }

    cudaMemcpy(res, gpuRes, sizeof(double), cudaMemcpyDeviceToHost);

    printf("GPU Result: %f\n", res[0]);
    printf("Time: %f s\n", ((double) clock() - gputime) / CLOCKS_PER_SEC);

    cudaFree(gpuVec);
    cudaFree(gpuRes);
    cudaFree(pinnedVec);
}

__global__ void vecSum_betterGPU(const double *a, double *gpuRes, const unsigned long n, int offset) {
    // dynamic shared memory size
    __shared__ double tmp[1024];

    // copy in shared memory
    unsigned long i = blockIdx.x * blockDim.x + threadIdx.x + offset*1024*1024;

    if (i < n) {
        tmp[threadIdx.x] = a[i];
    }

    __syncthreads();

    // do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            tmp[threadIdx.x] += tmp[threadIdx.x + s];
        }
        __syncthreads();
    }

    // last thread writes result
    if (threadIdx.x == 0) {
        gpuRes[blockIdx.x + offset * 1024] = tmp[0];
    }
}


__global__ void vecSum_GPU3(const double *a, double *res, const unsigned long n) {
    // dynamic shared memory size
    __shared__ double tmp[1024];

    // copy in shared memory
    unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        tmp[threadIdx.x] = a[i];
    }

    __syncthreads();

    // do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            tmp[threadIdx.x] += tmp[threadIdx.x + s];
        }
        __syncthreads();
    }

    // last thread writes result
    if (threadIdx.x == 0) {
        res[blockIdx.x] = tmp[0];
    }
}

void random_double(double *a, int n) {
    for (int i = 0; i < n; i++)
        a[i] = rand() % 10000; // random number between 0 and 9999
}
