#ifndef CUDA_ERROR_CHECK_H
#define CUDA_ERROR_CHECK_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "nccl.h"

#define cudaErrorCheck(call) do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define ncclErrorCheck(call) do { \
    ncclResult_t status = call; \
    if (status != ncclSuccess) { \
        fprintf(stderr, "NCCL Error at %s:%d: %s\n", \
            __FILE__, __LINE__, ncclGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define cublasErrorCheck(call) do { \
        cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error at %s:%d\n", \
            __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#endif 
