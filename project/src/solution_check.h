#include <stdio.h>
#include <stdlib.h>

void print_approx_solution(const double * matrix, size_t num_rows, int parallel, size_t num_cols = 1) {
    // Define the output file name
    const char* filename;
    if (parallel) {
        filename = "io/approx_solution_parallel.txt";
    } else {
        filename = "io/approx_solution_sequential.txt";
    }
    
    // Open the file for writing
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Cannot open output file %s\n", filename);
        return;
    }

    // Print the dimensions of the matrix to the file
    fprintf(file, "%zu %zu\n", num_rows, num_cols);

    // Iterate through the matrix and print each value
    for(size_t r = 0; r < num_rows; r++) {
        for(size_t c = 0; c < num_cols; c++) {
            double val = matrix[r * num_cols + c];
            fprintf(file, "%+6.5f ", val); // Use fprintf to write to the file
        }
        fprintf(file, "\n"); // Move to the next line after finishing a row
    }

    // Close the file
    fclose(file);
}