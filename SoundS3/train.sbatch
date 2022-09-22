#!/bin/bash

#SBATCH --mail-user=$USER@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:2080Ti:1
#SBATCH --job-name=sound
#SBATCH --output=slurm_train.out
#SBATCH --error=slurm_train.err
#SBATCH --partition=aquila,gpu

module purge
module load miniconda
source activate danielTorch
export PYTHONPATH="/scratch/$USER/pyPath:$PYTHONPATH"
cd "MajorScale_Sound_seq_DT-1-25_REP-2 (model 3)"
python main_train.py
