# Instructions to run the program

- On your PC (from /project):

To run just the parallel version (you need the cuda-toolkit), use: 

```bash NVCC_conjugategradient_compile_and_run.sh <size>```

To run both the sequential version and the parallel version, save an approximate solution (precision %+6.5f) and compare the results, use:

```bash runAndCompareSolutions.sh <size>```


- On the MeluXina supercomputer:

```sbatch submit.sh <size>```

In all cases, ensure that '<size>' corresponds to the dimensions of a matrix located in the 'io' folder.