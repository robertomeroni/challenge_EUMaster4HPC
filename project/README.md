# Instructions to run the program

For matrixes of size bigger than 65536, the program uses a NCCL implementation (multiple GPUs, communication directly from GPU memory to GPU memory with NVLink 3).

- On your PC (from /project):

If you are a normal person (you don't have a PC with 4 GPUs or more), the program will only work for sizes <= 65536 (if your GPU has enough memory).
To run the program you will need the cuBLAS library and the NCCL library.

- On the MeluXina supercomputer:

```sbatch submit.sh <size>```

Ensure that 'size' corresponds to the dimensions of a matrix located in the 'io' folder.