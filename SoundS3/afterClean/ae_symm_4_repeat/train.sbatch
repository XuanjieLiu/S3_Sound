#!/bin/bash

#SBATCH --mail-user=$USER@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:2080Ti:1
#SBATCH --job-name=a4r
#SBATCH --output=slurm_train.out
#SBATCH --error=slurm_train.err
#SBATCH --partition=aquila,gpu

module purge
module load miniconda
source activate danielTorch
python main_train.py
