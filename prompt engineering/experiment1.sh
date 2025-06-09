#!/bin/bash
#SBATCH --job-name=exp_1
#SBATCH --output=exp_1_%j.out
#SBATCH --error=exp_1_%j.err
#SBATCH --partition=gpu             # Or 'cpu' if you're not using a GPU
#SBATCH --gres=gpu:1           # Remove this line if using CPU only
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# Load and activate your conda environment
eval "$(/opt/software/anaconda3/2024.10-py3.12.7/bin/conda shell.bash hook)"
conda activate geollm

# Run python script
python experiment1.py

