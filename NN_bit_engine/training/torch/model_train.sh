#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=48:00:00

source /etc/profile.d/modules.sh
source /data03/home/kaungnyin/Anaconda/etc/profile.d/conda.sh

module load anaconda3 GPU/cuda/cuda-11.7

# 3. Activate and run
conda activate chessConda
/data03/home/kaungnyin/Anaconda/envs/chessConda/bin/python train.py