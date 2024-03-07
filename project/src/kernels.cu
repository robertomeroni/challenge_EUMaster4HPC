#include <cuda.h>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;
extern const int BLOCK_SIZE = 128;
const int WARP_SIZE = 32;


__global__ void apply_preconditioner(const double* r, double* z, const double* diagA, size_t size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        if (diagA[i] != 0) // check to avoid division by zero
            z[i] = r[i] / diagA[i];
        else
            z[i] = r[i]; // fallback in case of zero diagonal element
    }
}

__global__ void extract_diagonal(const double* d_A, double* diagA, size_t size)
{   
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        diagA[i] = d_A[i * size + i];
}

__global__ void nccl_extract_diagonal(const double* d_A, double* diagA, size_t num_rows, size_t size, int id)
{   
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_rows)
        diagA[i] = d_A[i * size + i + id * num_rows];
}

__global__ void nccl_extract_diagonal(const double* d_A, double* diagA, size_t num_rows, size_t size, size_t unused_rows, int id, size_t numGPUs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t global_row_index = i + id * num_rows; // calculate global row index in the original matrix

    // adjust for the last GPU which might have fewer rows
    if (id == numGPUs - 1) {
        num_rows -= unused_rows;
    }

    if (i < num_rows) {
        diagA[i] = d_A[i * size + global_row_index]; 
    }
}

__global__ void xpby(double *d_x, double beta, double *d_y, size_t size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        d_y[i] = d_x[i] + beta * d_y[i];
    }
}

// inizialization of the vectors x, r and p
__global__ void initialization(double *d_x, double *d_b, double *d_r, size_t size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        d_x[i] = 0.0;
        d_r[i] = d_b[i];
    }
}

__global__ void axpby(double alpha, const double *d_x, double beta, double *d_y, size_t size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        d_y[i] = alpha * d_x[i] + beta * d_y[i];
    }
}

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

// different versions of the dot product
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


// other versions of the dot product that I wrote, but are not used in the final implementation.
// the used version is the fastest one.
//
//
// __global__ void dot1(const double *d_x, const double *d_y, double *d_result, size_t size)
// {   
//     __shared__ double data[BLOCK_SIZE];
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     // reset d_result by the first thread of the first block
//     if (i == 0) {
//         *d_result = 0.0;
//     }

//     data[threadIdx.x] = i < size ? d_x[i] * d_y[i] : 0;
//     __syncthreads();
//     for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1)
//     {
//         if (threadIdx.x < stride)
//         {
//             data[threadIdx.x] += data[threadIdx.x + stride];
//         }
//         __syncthreads();
//     }
//     if (threadIdx.x == 0)
//     {
//        atomicAdd(d_result, data[0]);
//     }
// }

// __global__ void dot2(const double *d_x, const double *d_y, double *d_result, size_t size)
// {   
//     __shared__ double data[BLOCK_SIZE];
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     // reset d_result by the first thread of the first block
//     if (i == 0) {
//         *d_result = 0.0;
//     }
    
//     data[threadIdx.x] = i < size ? d_x[i] * d_y[i] : 0;
//         __syncthreads();
//         for (int stride = BLOCK_SIZE / 2; stride > 16; stride >>= 1)
//         {
//             if (threadIdx.x < stride)
//             {
//                 data[threadIdx.x] += data[threadIdx.x + stride];
//             }
//             __syncthreads();
//         }
//         if (threadIdx.x < WARP_SIZE)
//         {
//             data[threadIdx.x] = warpReduce(data[threadIdx.x]);
//         }
    
//     if (threadIdx.x == 0)
//     {
//        atomicAdd(d_result, data[0]);
//     }
// }
    
// __global__ void dot3(const double *d_x, const double *d_y, double *d_result, size_t size)
// {   
//     __shared__ double data[BLOCK_SIZE];
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     thread_block block = this_thread_block();

//     // reset d_result by the first thread of the first block
//     if (i == 0) {
//         *d_result = 0.0;
//     }
    
//     int tileSize = BLOCK_SIZE / WARP_SIZE;
//     thread_group g = tiled_partition(block, tileSize);
//     int id = g.thread_rank();

//     data[threadIdx.x] = i < size ? d_x[i] * d_y[i] : 0;
//     g.sync();
//     for (int stride = tileSize / 2; stride > 0; stride >>= 1)
//     {   
//         if (id < stride)
//         {
//             data[threadIdx.x] += data[threadIdx.x + stride];
//         }
//         g.sync();
//     }

//     if (id == 0)
//     {
//        atomicAdd(d_result, data[threadIdx.x]);
//     }
// }

// __global__ void dot4(const double *d_x, const double *d_y, double *d_result, size_t size)
// {   
//     __shared__ double data[BLOCK_SIZE];
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     thread_block block = this_thread_block();

//     // reset d_result by the first thread of the first block
//     if (i == 0) {
//         *d_result = 0.0;
//     }
    
//     int tileSize = BLOCK_SIZE / WARP_SIZE;
//     thread_group g = tiled_partition(block, tileSize);
//     int id = g.thread_rank();

//     // first operation and store in shared memory. 0 if i >= size (out of bounds access).
//     data[threadIdx.x] = i < size ? d_x[i] * d_y[i] : 0;
//     g.sync();
//     for (int stride = tileSize / 2; stride > 1; stride >>= 1)
//     {   
//         if (id < stride)
//         {
//             coalesced_group active = coalesced_threads();
//             data[threadIdx.x] += data[threadIdx.x + stride];
//             active.sync();
//         }
//     }
//     __syncthreads();
//     // store partial sums in the registers of the first warp (with last step of previous loop)
//     double reg = threadIdx.x < WARP_SIZE ? warpReduce(data[tileSize * threadIdx.x] + data[tileSize * threadIdx.x + 1]) : 0;

//     if (threadIdx.x == 0)
//     {
//         atomicAdd(d_result, reg);
//     }
// }

// __global__ void dot5(const double *d_x, const double *d_y, double *d_result, size_t size)
// {   
//     __shared__ double data[BLOCK_SIZE];
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     // reset d_result by the first thread of the first block
//     if (i == 0) {
//         *d_result = 0.0;
//     }
    
    
//     data[threadIdx.x] = i < size ? d_x[i] * d_y[i] : 0;
//     __syncthreads();
//     for (int stride = BLOCK_SIZE / 2; stride > WARP_SIZE; stride >>= 1)
//     {
//         if (threadIdx.x < stride)
//         {
//             data[threadIdx.x] += data[threadIdx.x + stride];
//         }
//         __syncthreads();
//     }
//     for (int stride = WARP_SIZE; stride > 0; stride >>= 1)
//     {
//         if (threadIdx.x < stride)
//         {
//             coalesced_group active = coalesced_threads();
//             data[threadIdx.x] += data[threadIdx.x + stride];
//             active.sync();
//         }
//     }
//     if (threadIdx.x == 0)
//     {
//        atomicAdd(d_result, data[0]);
//     }
// }



// __global__ void dot6(const double *d_x, const double *d_y, double *d_result, size_t size)
// {   
//     int i = blockIdx.x * blockDim.x + threadIdx.x;

//     // reset d_result by the first thread of the first block
//     if (i == 0) {
//         *d_result = 0.0;
//     }

//     double reg = i < size ? d_x[i] * d_y[i] : 0;
//     __syncthreads();
//     unsigned int mask = 0xffffffff; 

//     for (int offset = 16; offset > 0; offset /= 2) 
//     {
//         reg += __shfl_down_sync(mask, reg, offset);
//     }
    
//     if (threadIdx.x % WARP_SIZE == 0)
//     {
//         atomicAdd(d_result, reg);
//     }
// }







        