#!/bin/bash
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENMP_NUM_THREADS=1
export OMP_NUM_THREADS=1

for dataset in Spam Mushrooms Adult Credit 
do
    for problem in LR KM_16
    do
        for experiment in selection
        do
            python3 main.py --experiment=$experiment --dataset_name=$dataset --problem_name=$problem
        done
    done
done

