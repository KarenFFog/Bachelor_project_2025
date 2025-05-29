#!/bin/bash
#SBATCH --job-name=baseline_seed
#SBATCH --output=baseline_%j.out
#SBATCH --error=baseline_%j.err
#SBATCH --partition=gpu             # Or 'cpu' if you're not using a GPU
#SBATCH --gres=gpu:1                # Remove this line if using CPU only
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00

# Load and activate your conda environment
eval "$(/opt/software/anaconda3/2024.10-py3.12.7/bin/conda shell.bash hook)"
conda activate geollm

# Get the subset argument from command line
SUBSET=$1
SEED=$2

echo "Training baseline model on ${SUBSET}% subset (seed ${SEED})"

# Run your Python training script with argument
python run_baseline.py $SUBSET $SEED
