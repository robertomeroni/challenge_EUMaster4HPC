#!/bin/bash -l

#SBATCH --time=00:10:00
#SBATCH --account=p200301
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

PROGRAM="random_spd_system"
IO_FOLDER="io"
MATRIX="matrix.bin"
RHS="rhs.bin"
SIZE=2048

# check if the "io" folder exists, and create it if it doesn't
if [ ! -d "$IO_FOLDER" ]; then
    mkdir "$IO_FOLDER"
fi

# loading modules
module load OpenBLAS/0.3.20-GCC-11.3.0

# compile 
g++ -g -O2 src/${PROGRAM}.cpp -o ${PROGRAM} -lopenblas

# run 
./${PROGRAM} ${SIZE} ${IO_FOLDER}/${MATRIX} ${IO_FOLDER}/${RHS}

# clean the compiled program
rm -f ${PROGRAM}