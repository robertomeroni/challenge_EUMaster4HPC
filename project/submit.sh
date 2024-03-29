#!/bin/bash -l

# Submit a job to the Slurm job scheduler.
# Usage: sbatch submit.sh <size>


#SBATCH --time=01:00:00
#SBATCH --account=p200301
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=5


if [ "$#" -ne 1 ]; then
    echo "Usage: 'filename' <size>"
    exit 1
fi

SIZE=$1

PROGRAM="conjugate_gradients"
IO_FOLDER="io"
MATRIX="matrix_${SIZE}.bin"
RHS="rhs_${SIZE}.bin"
SOL="sol_${SIZE}.bin"

# check if the "io" folder exists, and create it if it doesn't
if [ ! -d "$IO_FOLDER" ]; then
    mkdir "$IO_FOLDER"
fi

# loading modules
module load env/release/2023.1
module load NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.2.0

# compile 
nvcc -arch=sm_80 src/${PROGRAM}.cu -o ${PROGRAM} -lnccl -lcublas 

# run 
./${PROGRAM} ${IO_FOLDER}/${MATRIX} ${IO_FOLDER}/${RHS} ${IO_FOLDER}/${SOL}

# clean the compiled program
rm -f ${PROGRAM}