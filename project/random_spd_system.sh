#!/bin/bash -l

#SBATCH --time=03:00:00
#SBATCH --account=p200301
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

PROGRAM="random_spd_system"
IO_FOLDER="io"
SIZE=131072
MATRIX="matrix_${SIZE}.bin"
RHS="rhs_${SIZE}.bin"


# check if the "io" folder exists, and create it if it doesn't
if [ ! -d "$IO_FOLDER" ]; then
    mkdir "$IO_FOLDER"
fi

# loading modules
module load env/release/2022.1
module load env/staging/2022.1
module load OpenBLAS/0.3.20-GCC-11.3.0

# compile 
g++ -g -O2 src/${PROGRAM}.cpp -o ${PROGRAM} -lopenblas

# run 
./${PROGRAM} ${SIZE} ${IO_FOLDER}/${MATRIX} ${IO_FOLDER}/${RHS}

# clean the compiled program
rm -f ${PROGRAM}