#!/bin/bash

module load suitesparse/5.1.2
module load nvhpc/22.5-nompi
module load cuda/11.1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/appl/SuiteSparse/5.1.2-sl73/lib:/appl/OpenBLAS/0.2.20/XeonGold6226R/gcc-6.4.0/lib