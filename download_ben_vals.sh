#!/bin/bash
#SBATCH --job-name=ben_val_dl
#SBATCH --output=ben_val_dl_%j.out
#SBATCH --error=ben_val_dl_%j.err
#SBATCH --partition=gpu       # or cpu if no GPU needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00

# Set up Conda environment
eval "$(/opt/software/anaconda3/2024.10-py3.12.7/bin/conda shell.bash hook)"
conda activate geollm

# Run Python script to download val split
python download_ben_val.py

