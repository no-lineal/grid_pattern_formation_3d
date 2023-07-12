#!/bin/bash
#SBATCH -J cube_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:4
#SBATCH -t 100:00:00
#SBATCH --mem=10G

source activate grid38

python -B main.py