#!/bin/bash
#SBATCH --job-name=mk_subsets
#SBATCH --output=mk_subsets_%j.out
#SBATCH --error=mk_subsets_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00

# Load and activate conda environment
eval "$(/opt/software/anaconda3/2024.10-py3.12.7/bin/conda shell.bash hook)"
conda activate geollm

# Run your Python training script with argument
python multilabel_stratification.py 
