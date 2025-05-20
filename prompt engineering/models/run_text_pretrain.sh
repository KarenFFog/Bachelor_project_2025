#!/bin/bash
#SBATCH --job-name=text_pretrain
#SBATCH --output=text_pretrain_%j.out
#SBATCH --error=text_pretrain_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00

# Load your environment
eval "$(/opt/software/anaconda3/2024.10-py3.12.7/bin/conda shell.bash hook)"
conda activate geollm

# Run the training script
python train_emb_image_model.py

