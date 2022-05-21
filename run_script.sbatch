#!/bin/bash
#SBATCH --mail-user=wangzh@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/wangzh/git/pytorch-cifar/out/%j.%N.stdout
#SBATCH --error=/home/wangzh/git/pytorch-cifar/out/%j.%N.stderr
#SBATCH --chdir=/home/wangzh/git/pytorch-cifar
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --account=pi-vveitch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --job-name=lab-gpu


i=${1}
maxiter=${2}

echo "python train_eval.py --i $i --maxiter $maxiter --run 1000"
echo which python
python -c 'import sys; print(sys.version_info[:])'
python train_eval.py --i $i --maxiter $maxiter --run 1000 
