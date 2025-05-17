#!/bin/bash
# The partition is the queue you want to run on. Standard is gpu and can be omitted.
#SBATCH -p gpu
#SBATCH --job-name=desc_gen
# Number of independent tasks we are going to start in this script
#SBATCH --ntasks=1
# Number of CPUs we want to allocate for each program
#SBATCH --cpus-per-task=4
# Amount of memory we want to allocate
#SBATCH --mem=12G
# Time limit
#SBATCH --time=07:00:00
# Request 1 GPU (required for model inference)
#SBATCH --gres=gpu:1

set -e

# Load Conda shell functions (for interactive + SLURM jobs)
eval "$(/opt/software/anaconda3/2024.10-py3.12.7/bin/conda shell.bash hook)"
conda activate geollm

# Run the generation script with the dataset split name passed as an argument
python pipeline_test_run.py $1

