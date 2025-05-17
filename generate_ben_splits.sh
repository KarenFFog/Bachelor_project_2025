#!/bin/bash
#SBATCH --job-name=ben_split_gen
#SBATCH --output=ben_split_gen_%j.out
#SBATCH --error=ben_split_gen_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1  # required if using the gpu partition

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

# Activate your environment
eval "$(/opt/software/anaconda3/2024.10-py3.12.7/bin/conda shell.bash hook)"
conda activate geollm

# Run the split generation script
python generate_ben_splits.py
