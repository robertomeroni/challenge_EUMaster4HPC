#!/bin/bash -l

#SBATCH --time=01:00:00
#SBATCH --account=p200301
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=5
#SBATCH --cpus-per-task=2

PROGRAM="conjugate_gradients"
IO_FOLDER="io"
MATRIX="matrix.bin"
RHS="rhs.bin"
SOL="sol.bin"

# check if the "io" folder exists, and create it if it doesn't
if [ ! -d "$IO_FOLDER" ]; then
    mkdir "$IO_FOLDER"
fi

# loading modules
module load CUDA/11.7.0

# compile 
nvcc -arch=sm_80 src/${PROGRAM}.cu -o ${PROGRAM} 

# run 
./${PROGRAM} ${IO_FOLDER}/${MATRIX} ${IO_FOLDER}/${RHS} ${IO_FOLDER}/${SOL}

# clean the compiled program
rm -f ${PROGRAM}