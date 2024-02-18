#include <cuda.h>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;
extern const int BLOCK_SIZE = 1024;

// inizialization of the vectors x, r and p
__global__ void initialization(double *d_x, double *d_b, double *d_r, double *d_p, size_t size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        d_x[i] = 0.0;
        d_r[i] = d_b[i];
        d_p[i] = d_b[i];
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
__global__ void dot1(const double *d_x, const double *d_y, double *d_result, size_t size)
{   
    __shared__ double data[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // reset d_result by the first thread of the first block
    if (i == 0) {
        *d_result = 0.0;
    }

    data[threadIdx.x] = d_x[i] * d_y[i];
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            data[threadIdx.x] += data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
       atomicAdd(d_result, data[0]);
    }
}

__global__ void dot2(const double *d_x, const double *d_y, double *d_result, size_t size)
{   
    __shared__ double data[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // reset d_result by the first thread of the first block
    if (i == 0) {
        *d_result = 0.0;
    }
    
        data[threadIdx.x] = d_x[i] * d_y[i];
        __syncthreads();
        for (int stride = BLOCK_SIZE / 2; stride > 16; stride >>= 1)
        {
            if (threadIdx.x < stride)
            {
                data[threadIdx.x] += data[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x < 32)
        {
            data[threadIdx.x] = warpReduce(data[threadIdx.x]);
        }
    
    if (threadIdx.x == 0)
    {
       atomicAdd(d_result, data[0]);
    }
}
    
__global__ void dot3(const double *d_x, const double *d_y, double *d_result, size_t size)
{   
    __shared__ double data[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    thread_block block = this_thread_block();

    // reset d_result by the first thread of the first block
    if (i == 0) {
        *d_result = 0.0;
    }
    
    int tileSize = 32;
    thread_group g = tiled_partition(block, tileSize);
    int id = g.thread_rank();
    if (i < size)
    {
        data[threadIdx.x] = d_x[i] * d_y[i];
        g.sync();
        for (int stride = tileSize / 2; stride > 0; stride >>= 1)
        {   
            if (threadIdx.x < BLOCK_SIZE - stride)
            {
                if (id < stride)
                {
                    data[threadIdx.x] += data[threadIdx.x + stride];
                }
            }
            g.sync();
        }
    }

    if (id == 0)
    {
       atomicAdd(d_result, data[threadIdx.x]);
    }
}

__global__ void dot4(const double *d_x, const double *d_y, double *d_result, size_t size)
{   
    __shared__ double data[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    thread_block block = this_thread_block();

    // reset d_result by the first thread of the first block
    if (i == 0) {
        *d_result = 0.0;
    }
    
    int tileSize = 32;
    thread_group g = tiled_partition(block, tileSize);
    int id = g.thread_rank();
    int groupId = threadIdx.x / tileSize;

    if (i < size)
    {
        data[threadIdx.x] = d_x[i] * d_y[i];
        g.sync();
        for (int stride = tileSize / 2; stride > 1; stride >>= 1)
        {   
            if (threadIdx.x < BLOCK_SIZE - stride)
            {
                if (id < stride)
                {
                    coalesced_group active = coalesced_threads();
                    data[threadIdx.x] += data[threadIdx.x + stride];
                    active.sync();
                }
            }
        }
        __syncthreads();
        double reg = groupId == 0 ? data[tileSize * id] + data[tileSize * id + 1] : 0;
        if (groupId == 0)
        {   
            reg = warpReduce(reg);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(d_result, reg);
        }
    }
}

__global__ void dot5(const double *d_x, const double *d_y, double *d_result, size_t size)
{   
    __shared__ double data[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // reset d_result by the first thread of the first block
    if (i == 0) {
        *d_result = 0.0;
    }
    
    if (i < size)
    {
        data[threadIdx.x] = d_x[i] * d_y[i];
        __syncthreads();
        for (int stride = BLOCK_SIZE / 2; stride > 32; stride >>= 1)
        {
            if (threadIdx.x < stride)
            {
                data[threadIdx.x] += data[threadIdx.x + stride];
            }
            __syncthreads();
        }
        for (int stride = 32; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
            {
                coalesced_group active = coalesced_threads();
                data[threadIdx.x] += data[threadIdx.x + stride];
                active.sync();
            }
        }
    }
    if (threadIdx.x == 0)
    {
       atomicAdd(d_result, data[0]);
    }
}

__global__ void dot6(const double *d_x, const double *d_y, double *d_result, size_t size)
{   
    __shared__ double data[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    thread_block block = this_thread_block();

    // reset d_result by the first thread of the first block
    if (i == 0) {
        *d_result = 0.0;
    }
    
    int tileSize = 32;
    thread_group g = tiled_partition(block, tileSize);
    int id = g.thread_rank();
    int groupId = threadIdx.x / tileSize;
    
    if (i < size)
    {
        data[threadIdx.x] = d_x[i] * d_y[i];
        g.sync();
        data[threadIdx.x] = warpReduce(data[threadIdx.x]);
        __syncthreads();
        double reg = groupId == 0 ? data[tileSize * id] : 0;
        if (groupId == 0)
        {   
            reg = warpReduce(reg);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(d_result, reg);
        }
    }
}



// __global__ void gemv(double alpha, const double *d_A, const double *d_x, double beta, double *d_y, size_t numRows, size_t numCols)
// {
//     // y = alpha * A * x + beta * y;
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     double y_val = 0.0;
//     if (i < numRows)
//     {
//         for(size_t c = 0; c < numCols; c++)
//         {
//             y_val += alpha * d_A[i * numCols + c] * d_x[c];
//         }
//         d_y[i] = beta * d_y[i] + y_val;
//     }
// }

__global__ void axpby(double alpha, const double *d_x, double beta, double *d_y, size_t size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        d_y[i] = alpha * d_x[i] + beta * d_y[i];
    }
}


        