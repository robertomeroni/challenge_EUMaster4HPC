#!/bin/bash

# -DAPPROX prints the solution in the file io/approx_solution.txt with precision %+6.5f
# Usage: $0 <size> <APPROX>
# APPROX is optional


if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <size> <APPROX> (optional)"
    exit 1
fi

SIZE=$1
APPROX=$2

if [ "$APPROX" == "APPROX" ]; then
    nvcc -arch=sm_80 src/conjugate_gradients.cu -o conjugate_gradients -lcublas -lnncl -DAPPROX
else
    nvcc -arch=sm_80 src/conjugate_gradients.cu -o conjugate_gradients -lcublas -lnncl
fi

./conjugate_gradients io/matrix_${SIZE}.bin io/rhs_${SIZE}.bin io/sol_${SIZE}.bin 