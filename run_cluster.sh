#!/bin/bash

maxiter=200
for i in {1..20}
do
    echo "sbatch run_script.sbatch  $i $maxiter"
    sbatch run_script.sbatch  $i $maxiter
done