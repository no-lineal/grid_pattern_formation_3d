#!/bin/bash
#SBATCH -J cube_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:15
#SBATCH -t 100:00:00
#SBATCH --mem=100G

source activate grid38

python -B main.py