#include <cuda.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <random>

using namespace cooperative_groups;
#define BLOCK_SIZE 256 // Adjust this value as necessary
#define WARP_SIZE 32 // Typically 32 for current architectures

// Use a utility macro to check CUDA errors
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (cudaSuccess != err) { \
            fprintf(stderr, "CUDA Error: %s: %d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", err, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// reduction within a warp of 32 threads, values are directly passed from register to register
__inline__ __device__ double warpReduce(double value) 
{
    unsigned int mask = 0xffffffff; 

    for (int offset = 16; offset > 0; offset /= 2) 
    {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value; 
}


__global__ void dot(const double *d_x, const double *d_y, double *d_result, size_t size)
{   
    __shared__ double data[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tileSize = BLOCK_SIZE / WARP_SIZE;
    thread_block block = this_thread_block();
    thread_group g = tiled_partition(block, tileSize);
    int id = g.thread_rank();

    // first operation and store in shared memory. 0 if i >= size (out of bounds access).
    data[threadIdx.x] = i < size ? d_x[i] * d_y[i] : 0;
    g.sync(); // for the for loop is only needed to synchronize between threads of the same group
    for (int stride = tileSize / 2; stride > 1; stride >>= 1)
    {   
        if (id < stride)
        {
            coalesced_group active = coalesced_threads();
            data[threadIdx.x] += data[threadIdx.x + stride];
            active.sync();
        }
    }
    __syncthreads();
    // store partial sums in the registers of the first warp (with last step of previous loop)
    double reg = threadIdx.x < WARP_SIZE ? data[tileSize * threadIdx.x] + data[tileSize * threadIdx.x + 1] : 0;

    // completely unrolled reduction in the first warp, shuffle directly from register to register
    if (threadIdx.x < WARP_SIZE)
    {
        reg = warpReduce(reg);
    }

    // atomic sum of the results of each block reduction
    if (threadIdx.x == 0)
    {
        atomicAdd(d_result, reg);
    }
}

// Main function
int main() {
    // Set the size of the vectors
    size_t size = 1 << 16;

    // Allocate memory and initialize data for x and y vectors
    double *x, *y, *result, *d_x, *d_y, *d_result;
    x = (double*)malloc(size * sizeof(double));
    y = (double*)malloc(size * sizeof(double));
    result = (double*)malloc(sizeof(double));
    CHECK_CUDA_ERROR(cudaMalloc(&d_x, size * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_y, size * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, sizeof(double)));

    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::mt19937 gen;

    // Initialize x and y with some values
    for (size_t i = 0; i < size; i++) {
        x[i] = dis(gen);
        y[i] = dis(gen);
    }

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_y, y, size * sizeof(double), cudaMemcpyHostToDevice));

    // Clear the result
    *result = 0.0;
    CHECK_CUDA_ERROR(cudaMemcpy(d_result, result, sizeof(double), cudaMemcpyHostToDevice));

    // Define the number of blocks and threads
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Define CUDA events
    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA_ERROR(cudaEventCreate(&startEvent));
    CHECK_CUDA_ERROR(cudaEventCreate(&stopEvent));
    float milliseconds = 0;
    
    // Create a handle for cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    double cublas_result;

    // Warm-up phase for cuBLAS
    double warmup_result; // Temporary variable to store the result
    for (int i = 0; i < 10; ++i) { // Execute the operation multiple times
        cublasDdot(handle, size, d_x, 1, d_y, 1, &warmup_result);
    }
    cudaDeviceSynchronize(); // Ensure all operations are complete
    
    // Measure the execution time of the cuBLAS dot product using CUDA events
    CHECK_CUDA_ERROR(cudaEventRecord(startEvent));
    cublasDdot(handle, size, d_x, 1, d_y, 1, &cublas_result);
    cudaDeviceSynchronize(); // Wait for the cuBLAS operation to complete
    CHECK_CUDA_ERROR(cudaEventRecord(stopEvent));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stopEvent));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));
    std::cout << "cuBLAS dot product execution time: " << milliseconds << " ms\n";


    // Warm-up phase for the custom dot product kernel
    for (int i = 0; i < 10; ++i) { // Execute the kernel multiple times
        dot<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_result, size);
    }
    cudaMemset(d_result, 0, sizeof(double)); // Clear the result
    cudaDeviceSynchronize(); // Ensure all operations are complete

    // Measure the execution time of the custom dot product kernel using CUDA events
    CHECK_CUDA_ERROR(cudaEventRecord(startEvent));
    dot<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_result, size);
    cudaDeviceSynchronize(); // Wait for the kernel to complete
    CHECK_CUDA_ERROR(cudaEventRecord(stopEvent));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stopEvent));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));
    std::cout << "Custom dot product execution time: " << milliseconds << " ms\n";

      // Retrieve result for the custom dot product
    CHECK_CUDA_ERROR(cudaMemcpy(result, d_result, sizeof(double), cudaMemcpyDeviceToHost));

    // Compare the results with a tolerance since we are dealing with floating point numbers
    double tolerance = 1e-5; 
    if (fabs(cublas_result - *result) <= tolerance) {
        std::cout << "Results are the same within tolerance." << std::endl;
    } else {
        std::cout << "Results differ: cuBLAS result = " << cublas_result << ", Custom dot product result = " << *result << std::endl;
    }


    // Clean up
    free(x);
    free(y);
    free(result);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
    CHECK_CUDA_ERROR(cudaEventDestroy(startEvent));
    CHECK_CUDA_ERROR(cudaEventDestroy(stopEvent));

    return 0;
}