#!/bin/sh

#SBATCH --partition=GPUQ
#SBATCH --job-name=medseg-SwinUNETR
#SBATCH --account=share-ie-idi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=(a100)
#SBATCH --output=logs/out-%j.log
#SBATCH --error=logs/out-%j.log


cd /cluster/work/andrebw/medical-image-segmentation
pwd
. venv/bin/activate
python train_basic.py
