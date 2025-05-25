#!/bin/bash
#SBATCH --job-name=eval_baseline
#SBATCH --output=eval_baseline_%j.out
#SBATCH --error=eval_baseline_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00

# Activate your conda environment
eval "$(/opt/software/anaconda3/2024.10-py3.12.7/bin/conda shell.bash hook)"
conda activate geollm

# Get the subset argument from command line
SUBSET=$1

echo "Running eval on baseline: $SUBSET pct"

# Run Python evaluation script with argument
python evaluate_baseline.py $SUBSET

