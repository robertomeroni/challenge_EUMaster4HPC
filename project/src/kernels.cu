#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;
namespace cg = cooperative_groups;
extern const int BLOCK_SIZE = 1024;

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

__global__ void dot(const double *d_x, const double *d_y, double *d_result, size_t size)
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

        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
            {
                data[threadIdx.x] += data[threadIdx.x + stride];
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0)
    {
       atomicAdd(d_result, data[0]);
    }
}

__global__ void gemv(double alpha, const double *d_A, const double *d_x, double beta, double *d_y, size_t numRows, size_t numCols)
{
    // y = alpha * A * x + beta * y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double y_val = 0.0;
    if (i < numRows)
    {
        for(size_t c = 0; c < numCols; c++)
        {
            y_val += alpha * d_A[i * numCols + c] * d_x[c];
        }
        d_y[i] = beta * d_y[i] + y_val;
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


        