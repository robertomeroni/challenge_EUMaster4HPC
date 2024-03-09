#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <time.h>
#include <cuda.h>
#include <nccl.h>
#include <iostream>
#include "cublas_v2.h"
#include "kernels.cu"
#include "cuda_error_check.h"
#include "solution_check.h"

clock_t start, end;
double cpu_time_used;

bool read_matrix_from_file(const char * filename, double ** matrix_out, size_t * num_rows_out, size_t * num_cols_out)
{
    double * matrix;
    size_t num_rows;
    size_t num_cols;

    FILE * file = fopen(filename, "rb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fread(&num_rows, sizeof(size_t), 1, file);
    fread(&num_cols, sizeof(size_t), 1, file);
    matrix = new double[num_rows * num_cols];
    fread(matrix, sizeof(double), num_rows * num_cols, file);

    *matrix_out = matrix;
    *num_rows_out = num_rows;
    *num_cols_out = num_cols;

    fclose(file);

    return true;
}

bool write_matrix_to_file(const char * filename, const double * matrix, size_t num_rows, size_t num_cols)
{
    FILE * file = fopen(filename, "wb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fwrite(&num_rows, sizeof(size_t), 1, file);
    fwrite(&num_cols, sizeof(size_t), 1, file);
    fwrite(matrix, sizeof(double), num_rows * num_cols, file);

    fclose(file);

    return true;
}

void print_matrix(const double * matrix, size_t num_rows, size_t num_cols, FILE * file = stdout)
{
    fprintf(file, "%zu %zu\n", num_rows, num_cols);
    for(size_t r = 0; r < num_rows; r++)
    {
        for(size_t c = 0; c < num_cols; c++)
        {
            double val = matrix[r * num_cols + c];
            printf("%+6.3f ", val);
        }
        printf("\n");
    }
}



void nccl_conjugate_gradients(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error)
{   
    // NCCL initialization
    const int numGPUs = 4;
    int gpus[numGPUs] = {0, 1, 2, 3}; 
    ncclComm_t comms[numGPUs];
    cudaStream_t stream1[numGPUs];
    cudaStream_t stream2[numGPUs];

    bool not_divisible = size % numGPUs != 0;
    size_t num_rows = size / numGPUs + (not_divisible);
    int unused_rows = num_rows * numGPUs - size;

    double alpha, beta, bb, rr, rr_new;
    double pAp;
    const double gemv_alpha = 1.0;
    const double gemv_beta = 0.0;
    double **d_p = (double **) malloc(numGPUs * sizeof(double *));
    double **d_A = (double **) malloc(numGPUs * sizeof(double *));
    double **d_Ap = (double **) malloc(numGPUs * sizeof(double *));
    double * d_diagA[numGPUs];
    double * d_x;
    double * d_b;
    double * d_r;
    double * d_z;
    double * d_bb;
    double * d_rr_new;
    double * d_pAp;
    int num_iters;

    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int numThreads = BLOCK_SIZE;

    ncclErrorCheck(ncclCommInitAll(comms, numGPUs, gpus));

     // memory allocation on the GPU.
    cudaErrorCheck(cudaMalloc((void**)&d_x, size * sizeof(double)));
    cudaErrorCheck(cudaMalloc((void**)&d_b, size * sizeof(double)));
    cudaErrorCheck(cudaMalloc((void**)&d_r, size * sizeof(double)));
    cudaErrorCheck(cudaMalloc((void**)&d_z, size * sizeof(double)));
    cudaErrorCheck(cudaMalloc((void**)&d_bb, sizeof(double)));
    cudaErrorCheck(cudaMalloc((void**)&d_rr_new, sizeof(double)));
    cudaErrorCheck(cudaMalloc((void**)&d_pAp, sizeof(double)));

    for(int i = 0; i < numGPUs; i++)
    {   
        cudaErrorCheck(cudaSetDevice(gpus[i]));
        cudaErrorCheck(cudaStreamCreate(&stream1[i]));
        cudaErrorCheck(cudaStreamCreate(&stream2[i]));
        // memory allocation on the GPU.
        cudaErrorCheck(cudaMalloc(d_p + i, size * sizeof(double)));
        cudaErrorCheck(cudaMalloc(d_A + i, num_rows * size * sizeof(double)));
        cudaErrorCheck(cudaMalloc(d_Ap + i, numGPUs * num_rows * sizeof(double)));
        cudaErrorCheck(cudaMalloc(d_diagA + i, size * sizeof(double)));

        // each GPU gets a subset of the matrix A
        if (i < numGPUs - 1)
        {
            cudaErrorCheck(cudaMemcpyAsync(d_A[i], A + i * num_rows * size, num_rows * size * sizeof(double), cudaMemcpyHostToDevice, stream2[i]));
        }
        else
        {
            cudaErrorCheck(cudaMemcpyAsync(d_A[i], A + i * num_rows * size, (num_rows - unused_rows) * size * sizeof(double), cudaMemcpyHostToDevice, stream2[i]));
        }
    }   

    // if the matrix is not divisible by 4, fill the empty rows from last GPU with zeros
    if (not_divisible)
    {
        cudaErrorCheck(cudaMemsetAsync(d_A[numGPUs - 1] + (num_rows - unused_rows) * size, 0, unused_rows * size * sizeof(double), stream1[numGPUs - 1]));
    }
    
    cudaErrorCheck(cudaSetDevice(gpus[0]));
    cudaErrorCheck(cudaMemcpy(d_b, b, size * sizeof(double), cudaMemcpyHostToDevice));

    dot <<<numBlocks, numThreads>>> (d_b, d_b, d_bb, size);
    for (int i = 0; i < numGPUs; i++)
    {
        cudaSetDevice(gpus[i]);
        nccl_extract_diagonal <<<numBlocks, numThreads, 0, stream2[i]>>> (d_A[i], d_diagA[i] + i * num_rows, num_rows, size, unused_rows, i, numGPUs);
    }
    
    ncclGroupStart();
    for(int i = 0; i < numGPUs; i++) 
    {
        cudaErrorCheck(cudaSetDevice(gpus[i]));
        ncclErrorCheck(ncclAllGather(d_diagA[i] + i * num_rows, d_diagA[i], num_rows, ncclDouble, comms[i], stream2[i]));
    }
    ncclGroupEnd();
    cudaSetDevice(gpus[0]);
    apply_preconditioner <<<numBlocks, numThreads, 0, stream2[0]>>> (d_b, d_z, d_diagA[0], size);
    dot <<<numBlocks, numThreads, 0, stream2[0]>>> (d_b, d_z, d_rr_new, size);
    initialization <<<numBlocks, numThreads, 0, stream1[0]>>> (d_x, d_b, d_r, size);
    cudaErrorCheck(cudaMemcpyAsync(d_p[0], d_z, size * sizeof(double), cudaMemcpyDeviceToDevice, stream2[0]));
    cudaErrorCheck(cudaMemcpyAsync(&rr, d_rr_new, sizeof(double), cudaMemcpyDeviceToHost,stream2[0]));
    cudaErrorCheck(cudaMemcpy(&bb, d_bb, sizeof(double), cudaMemcpyDeviceToHost));

    cublasHandle_t handle[numGPUs];
    for(int i = 0; i < numGPUs; i++)
    {
        cudaErrorCheck(cudaSetDevice(gpus[i]));
        cublasErrorCheck(cublasCreate(&handle[i]));
        cublasErrorCheck(cublasSetStream(handle[i], stream1[i])); // link the cuda stream to the cublas handle
    }

    for (int i = 0; i < numGPUs; i++)
    {
        cudaSetDevice(gpus[i]);
        cudaStreamSynchronize(stream2[i]);
    }
    cudaErrorCheck(cudaSetDevice(gpus[0]));


    // MAIN LOOP
    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {   
        // broadcast d_p to all GPUs
        ncclGroupStart();
        for(int i = 0; i < numGPUs; i++) 
        {
            cudaErrorCheck(cudaSetDevice(gpus[i]));
            ncclErrorCheck(ncclBroadcast(d_p[0], d_p[i], size, ncclDouble, 0, comms[i], stream1[i]));
        }
        ncclGroupEnd();

        // each GPU computes gemv on a subset of the matrix A
        for (int i = 0; i < numGPUs; i++)
        {   
            cudaSetDevice(gpus[i]);
            cublasErrorCheck(cublasDgemv(handle[i], CUBLAS_OP_T, size, num_rows, &gemv_alpha, d_A[i], size, d_p[i], 1, &gemv_beta, d_Ap[i] + i * num_rows, 1));
        }
        
        // allgather the results of gemv
        ncclErrorCheck(ncclGroupStart());   
        for (int i = 0; i < numGPUs; i++)
        {
            cudaSetDevice(gpus[i]);
            ncclErrorCheck(ncclAllGather(d_Ap[i] + i * num_rows, d_Ap[i], num_rows, ncclDouble, comms[i], stream1[i]));
        }
        ncclErrorCheck(ncclGroupEnd());

        // synchronize over all GPUs
        for (int i = 0; i < numGPUs; i++)
        {
            cudaSetDevice(gpus[i]);
            cudaStreamSynchronize(stream1[i]);
        }
        cudaErrorCheck(cudaSetDevice(gpus[0]));

        dot <<<numBlocks, numThreads>>> (d_p[0], d_Ap[0], d_pAp, size);
        cudaErrorCheck(cudaMemcpy(&pAp, d_pAp, sizeof(double), cudaMemcpyDeviceToHost));
        alpha = rr / pAp;
        axpby <<<numBlocks, numThreads>>> (-alpha, d_Ap[0], 1.0, d_r, size); 
        cudaMemsetAsync(d_pAp, 0, sizeof(double)); // reset dot product to zero, done in parallel with stream2
        cudaMemsetAsync(d_rr_new, 0, sizeof(double));
        cudaStreamSynchronize(stream2[0]); // ensure that axbpy on x from the previous iteration has terminated
        axpby <<<numBlocks, numThreads, 0, stream2[0]>>> (alpha, d_p[0], 1.0, d_x, size); // x is not needed until the next iteration and is only get called by this kernel
        apply_preconditioner <<<numBlocks, numThreads>>> (d_r, d_z, d_diagA[0], size);
        dot <<<numBlocks, numThreads>>> (d_r, d_z, d_rr_new, size);
        cudaErrorCheck(cudaMemcpy(&rr_new, d_rr_new, sizeof(double), cudaMemcpyDeviceToHost));
        beta = rr_new / rr;
        xpby <<<numBlocks, numThreads>>> (d_z, beta, d_p[0], size); // this can be done after beta is calculated
        rr = rr_new;
        if(std::sqrt(rr / bb) < rel_error) { break; }
    }
    
    // copy the solution back to the host
    cudaErrorCheck(cudaMemcpyAsync(x, d_x, size * sizeof(double), cudaMemcpyDeviceToHost,stream2[0]));

    // cleaning up
    cudaErrorCheck(cudaFree(d_x));
    cudaErrorCheck(cudaFree(d_b));
    cudaErrorCheck(cudaFree(d_r));
    cudaErrorCheck(cudaFree(d_bb));
    cudaErrorCheck(cudaFree(d_pAp));
    cudaErrorCheck(cudaFree(d_rr_new));
    cudaErrorCheck(cudaFree(d_z));
    for(int i = 0; i < numGPUs; i++)
    {   
        cudaErrorCheck(cudaFree(d_p[i]));
        cudaErrorCheck(cudaFree(d_A[i]));
        cudaErrorCheck(cudaFree(d_Ap[i]));
        cudaErrorCheck(cudaFree(d_diagA[i]));
        cudaErrorCheck(cudaStreamDestroy(stream1[i]));
        cudaErrorCheck(cudaStreamDestroy(stream2[i]));
        ncclErrorCheck(ncclCommDestroy(comms[i]));
        cublasErrorCheck(cublasDestroy(handle[i]));
    }

    if(num_iters <= max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
    }
}





void conjugate_gradients(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error)
{   
    double alpha, beta, bb, rr, rr_new;
    double pAp;
    const double gemv_alpha = 1.0;
    const double gemv_beta = 0.0;
    double * d_A = new double[size * size];
    double * d_diagA = new double[size];
    double * d_Ap = new double[size];
    double * d_x = new double[size];
    double * d_b = new double[size];
    double * d_r = new double[size];
    double * d_p = new double[size];
    double * d_z = new double[size];
    double * d_bb = new double;
    double * d_rr_new = new double;
    double * d_pAp = new double;
    int num_iters;

    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int numThreads = BLOCK_SIZE;
    
    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    cublasHandle_t handle;
    cublasCreate(&handle);

    // memory allocation on the GPU.
    cudaErrorCheck(cudaMalloc((void**)&d_x, size * sizeof(double)));
    cudaErrorCheck(cudaMalloc((void**)&d_b, size * sizeof(double)));
    cudaErrorCheck(cudaMalloc((void**)&d_r, size * sizeof(double)));
    cudaErrorCheck(cudaMalloc((void**)&d_p, size * sizeof(double)));
    cudaErrorCheck(cudaMalloc((void**)&d_z, size * sizeof(double)));
    cudaErrorCheck(cudaMalloc((void**)&d_bb, sizeof(double)));
    cudaErrorCheck(cudaMalloc((void**)&d_rr_new, sizeof(double)));
    cudaErrorCheck(cudaMalloc((void**)&d_A, size * size * sizeof(double)));
    cudaErrorCheck(cudaMalloc((void**)&d_diagA, size * sizeof(double)));
    cudaErrorCheck(cudaMalloc((void**)&d_Ap, size * sizeof(double)));
    cudaErrorCheck(cudaMalloc((void**)&d_pAp, sizeof(double)));

    // memory copy from the CPU to the GPU, copy of matrix A is done in parallel with initialization and dot product.
    cudaErrorCheck(cudaMemcpyAsync(d_A, A, size * size * sizeof(double), cudaMemcpyHostToDevice, stream1));
    cudaErrorCheck(cudaMemcpy(d_b, b, size * sizeof(double), cudaMemcpyHostToDevice));
    
    dot <<<numBlocks, numThreads>>> (d_b, d_b, d_bb, size);
    extract_diagonal <<<numBlocks, numThreads, 0, stream1>>> (d_A, d_diagA, size);
    apply_preconditioner <<<numBlocks, numThreads, 0, stream1>>> (d_b, d_z, d_diagA, size);
    initialization <<<numBlocks, numThreads, 0, stream2>>> (d_x, d_b, d_r, size);
    cudaErrorCheck(cudaMemcpyAsync(d_p, d_z, size * sizeof(double), cudaMemcpyDeviceToDevice, stream1));
    cudaErrorCheck(cudaMemcpyAsync(&bb, d_bb, sizeof(double), cudaMemcpyDeviceToHost,stream3));
    dot <<<numBlocks, numThreads, 0, stream1>>> (d_b, d_z, d_rr_new, size); 
    cudaErrorCheck(cudaMemcpy(&rr, d_rr_new, sizeof(double), cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaDeviceSynchronize());

    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        cublasDgemv(handle, CUBLAS_OP_N, size, size, &gemv_alpha, d_A, size, d_p, 1, &gemv_beta, d_Ap, 1);
        dot <<<numBlocks, numThreads>>> (d_p, d_Ap, d_pAp, size);
        cudaErrorCheck(cudaMemcpy(&pAp, d_pAp, sizeof(double), cudaMemcpyDeviceToHost));
        alpha = rr / pAp;
        axpby <<<numBlocks, numThreads>>> (-alpha, d_Ap, 1.0, d_r, size); 
        cudaMemsetAsync(d_pAp, 0, sizeof(double)); // reset dot product to zero, done in parallel with stream1
        cudaMemsetAsync(d_rr_new, 0, sizeof(double));
        cudaStreamSynchronize(stream1); // ensure that axbpy on x from the previous iteration has terminated
        axpby <<<numBlocks, numThreads, 0, stream1>>> (alpha, d_p, 1.0, d_x, size); // x is not needed until the next iteration and is only get called by this kernel
        apply_preconditioner <<<numBlocks, numThreads>>> (d_r, d_z, d_diagA, size);
        dot <<<numBlocks, numThreads>>> (d_r, d_z, d_rr_new, size);
        cudaErrorCheck(cudaMemcpy(&rr_new, d_rr_new, sizeof(double), cudaMemcpyDeviceToHost));
        beta = rr_new / rr;
        xpby <<<numBlocks, numThreads>>> (d_z, beta, d_p, size); // this can be done after beta is calculated
        rr = rr_new;
        if(std::sqrt(rr / bb) < rel_error) { break; }
    }

    cudaErrorCheck(cudaMemcpyAsync(x, d_x, size * sizeof(double), cudaMemcpyDeviceToHost,stream1));

    // cleaning up
    cudaErrorCheck(cudaFree(d_x));
    cudaErrorCheck(cudaFree(d_b));
    cudaErrorCheck(cudaFree(d_r));
    cudaErrorCheck(cudaFree(d_p));
    cudaErrorCheck(cudaFree(d_z));
    cudaErrorCheck(cudaFree(d_A));
    cudaErrorCheck(cudaFree(d_diagA));
    cudaErrorCheck(cudaFree(d_Ap));
    cudaErrorCheck(cudaFree(d_bb));
    cudaErrorCheck(cudaFree(d_pAp));
    cudaErrorCheck(cudaFree(d_rr_new));
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cublasDestroy(handle);

    if(num_iters <= max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
    }
}



int main(int argc, char ** argv)
{
    printf("Usage: ./random_matrix input_file_matrix.bin input_file_rhs.bin output_file_sol.bin max_iters rel_error\n");
    printf("All parameters are optional and have default values\n");
    printf("\n");

    const char * input_file_matrix = "io/matrix.bin";
    const char * input_file_rhs = "io/rhs.bin";
    const char * output_file_sol = "io/sol.bin";
    int max_iters = 1000;
    double rel_error = 1e-9;

    if(argc > 1) input_file_matrix = argv[1];
    if(argc > 2) input_file_rhs = argv[2];
    if(argc > 3) output_file_sol = argv[3];
    if(argc > 4) max_iters = atoi(argv[4]);
    if(argc > 5) rel_error = atof(argv[5]);

    printf("Command line arguments:\n");
    printf("  input_file_matrix: %s\n", input_file_matrix);
    printf("  input_file_rhs:    %s\n", input_file_rhs);
    printf("  output_file_sol:   %s\n", output_file_sol);
    printf("  max_iters:         %d\n", max_iters);
    printf("  rel_error:         %e\n", rel_error);
    printf("\n");



    double * matrix;
    double * rhs;
    size_t size;

    {
        printf("Reading matrix from file ...\n");
        size_t matrix_rows;
        size_t matrix_cols;
        bool success_read_matrix = read_matrix_from_file(input_file_matrix, &matrix, &matrix_rows, &matrix_cols);
        if(!success_read_matrix)
        {
            fprintf(stderr, "Failed to read matrix\n");
            return 1;
        }
        printf("Done\n");
        printf("\n");

        printf("Reading right hand side from file ...\n");
        size_t rhs_rows;
        size_t rhs_cols;
        bool success_read_rhs = read_matrix_from_file(input_file_rhs, &rhs, &rhs_rows, &rhs_cols);
        if(!success_read_rhs)
        {
            fprintf(stderr, "Failed to read right hand side\n");
            return 2;
        }
        printf("Done\n");
        printf("\n");

        if(matrix_rows != matrix_cols)
        {
            fprintf(stderr, "Matrix has to be square\n");
            return 3;
        }
        if(rhs_rows != matrix_rows)
        {
            fprintf(stderr, "Size of right hand side does not match the matrix\n");
            return 4;
        }
        if(rhs_cols != 1)
        {
            fprintf(stderr, "Right hand side has to have just a single column\n");
            return 5;
        }

        size = matrix_rows;
    }

    printf("Solving the system ...\n");
    double * sol = new double[size];

    start = clock();

    // if the matrix is big use the implementation with NCCL (multiple GPUs, each with a subset of the matrix A)
    if (size <= 65536)
        conjugate_gradients(matrix, rhs, sol, size, max_iters, rel_error);
    else
        nccl_conjugate_gradients(matrix, rhs, sol, size, max_iters, rel_error);

    end = clock();

    printf("Done\n");
    printf("\n");

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Execution time: %f seconds\n", cpu_time_used);
    printf("\n");

    printf("Writing solution to file ...\n");
    bool success_write_sol = write_matrix_to_file(output_file_sol, sol, size, 1);
    if(!success_write_sol)
    {
        fprintf(stderr, "Failed to save solution\n");
        return 6;
    }
    printf("Done\n");
    printf("\n");

    #ifdef APPROX
    print_approx_solution(sol, size, 1);
    #endif

    delete[] matrix;
    delete[] rhs;
    delete[] sol;

    printf("Finished successfully\n");

    return 0;
}
