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
    g++ -g -O2 src/conjugate_gradients.cpp -o conjugate_gradients -w -DAPPROX
else
    g++ -g -O2 src/conjugate_gradients.cpp -o conjugate_gradients -w 
fi

./conjugate_gradients io/matrix_${SIZE}.bin io/rhs_${SIZE}.bin io/sol_${SIZE}.bin