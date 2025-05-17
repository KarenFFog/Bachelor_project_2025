#!/bin/bash
#SBATCH --job-name=baseline_train
#SBATCH --output=baseline_train_%j.out
#SBATCH --error=baseline_train_%j.err
#SBATCH --partition=gpu             # Or 'cpu' if you're not using a GPU
#SBATCH --gres=gpu:1                # Remove this line if using CPU only
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# Load and activate your conda environment
eval "$(/opt/software/anaconda3/2024.10-py3.12.7/bin/conda shell.bash hook)"
conda activate geollm

# Run your Python training script
python run_baseline.py

