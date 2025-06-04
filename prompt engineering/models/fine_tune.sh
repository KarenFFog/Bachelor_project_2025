#!/bin/bash
#SBATCH --job-name=fine_t
#SBATCH --output=fine_t_%j.out
#SBATCH --error=fine_t_%j.err
#SBATCH --partition=gpu             # Or 'cpu' if you're not using a GPU
#SBATCH --gres=gpu:titanrtx:1           # Remove this line if using CPU only
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

# Load and activate your conda environment
eval "$(/opt/software/anaconda3/2024.10-py3.12.7/bin/conda shell.bash hook)"
conda activate geollm

# Get the subset argument from command line
SUBSET=$1
SEED=$2

echo "Training ft model on ${SUBSET}% subset (seed ${SEED})"

# Run python script, pass subset percentage (1, 5, 10 or 100)
python fine_tune.py $SUBSET $SEED
