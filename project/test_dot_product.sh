#!/bin/bash -l

# Submit a job to the Slurm job scheduler.
# Usage: sbatch test_dot_product.sh <number of runs>


#SBATCH --time=01:00:00
#SBATCH --account=p200301
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=5


if [ "$#" -ne 1 ]; then
    echo "Usage: 'filename' <number of runs>"
    exit 1
fi

NUM_RUNS=$1

PROGRAM="dotTest"

# loading modules
module load env/release/2023.1
module load NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.2.0

# compile 
nvcc -arch=sm_80 src/${PROGRAM}.cu -o ${PROGRAM} -lcublas

# run
for (( i=1; i<=NUM_RUNS; i++ ))
do
    echo "Run $i"
    ./${PROGRAM}
done