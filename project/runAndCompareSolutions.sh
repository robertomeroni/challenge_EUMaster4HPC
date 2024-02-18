#!/bin/bash

# This script executes both the sequential and the parallel version of the program, prints the approximate solution and compares the results.
# It requires the CUDA toolkit, cuBLAS in particular.
# Usage: $0 <size>

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <size>"
    exit 1
fi

SIZE=$1

# compile and run the programs
bash GCC_conjugategradient_compile_and_run.sh ${SIZE} APPROX
bash NVCC_conjugategradient_compile_and_run.sh ${SIZE} APPROX

echo
echo

# compare solutions
diff io/approx_solution_parallel.txt io/approx_solution_sequential.txt > /dev/null && echo -e "\033[1;32mThe approximate solution is correct\033[0m"  || echo -e "\033[1;31mThe solution is wrong\033[0m" 
echo