#!/bin/bash
#SBATCH --job-name=gpu_job # name job
#SBATCH --partition=gpu # use GPU queue
#SBATCH --gres=gpu:2 # request 2 GPU
#SBATCH --cpus-per-task=4 # adjust CPU cores
#SBATCH --mem=32G # allocate 32G of memory
#SBATCH --time=03:00:00 # runtime max of 2 hours

module load anaconda3
module load cuda/11.8 # or whatever CUDA version matches your packages

conda activate chessConda # or your env name
python test_env.py
