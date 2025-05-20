#!/bin/bash
#SBATCH --job-name=embed_sbert
#SBATCH --output=embed_sbert_%j.out
#SBATCH --error=embed_sbert_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# Activate your conda environment
eval "$(/opt/software/anaconda3/2024.10-py3.12.7/bin/conda shell.bash hook)"
conda activate geollm

# Run the embedding script
python embed_descriptions.py

