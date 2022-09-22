#!/bin/bash

#SBATCH --mail-user=daniel.chin@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:2080Ti:1
#SBATCH --job-name=v4n
#SBATCH --output=slurm_train.out
#SBATCH --error=slurm_train.err
#SBATCH --partition=aquila,gpu

module purge
module load miniconda
source activate danielTorch
python main_train.py
