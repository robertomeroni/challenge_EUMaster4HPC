#ifndef CUDA_ERROR_CHECK_H
#define CUDA_ERROR_CHECK_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define cudaErrorCheck(call) do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#endif // CUDA_ERROR_CHECK_H
