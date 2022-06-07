#!/bin/bash
#SBATCH --job-name=vgg13_exp1
#SBATCH --output=vgg13_exp1_%A.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=96:00:00
#SBATCH --partition=gpu

module purge
module load CUDA/11.1.1-GCC-10.2.0
module load cuDNN/8.0.5.39-CUDA-11.1.1
module load CUDAcore/11.1.1

python3 experiments.py 1
