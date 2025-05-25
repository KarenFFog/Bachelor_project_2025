#!/bin/bash
#SBATCH --job-name=lin_prob
#SBATCH --output=lin_prob_%j.out
#SBATCH --error=lin_prob_%j.err
#SBATCH --partition=gpu             # Or 'cpu' if you're not using a GPU
#SBATCH --gres=gpu:a100:1       # Remove this line if using CPU only
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# Load and activate your conda environment
eval "$(/opt/software/anaconda3/2024.10-py3.12.7/bin/conda shell.bash hook)"
conda activate geollm

# Get the subset argument from command line
SUBSET=$1

echo "Running training with subset: $SUBSET"

# Run python script, pass subset percentage (1, 5, 10 or 100)
python linear_prob.py $SUBSET
