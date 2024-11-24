#!/bin/sh

#SBATCH --partition=GPUQ
#SBATCH --job-name=medseg-SegResNet-lower-lr
#SBATCH --account=share-ie-idi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=(a100)
#SBATCH --output=logs/idun/out-%j.log
#SBATCH --error=logs/idun/out-%j.log

echo "%x-%j"
cd /cluster/work/andrebw/medical-image-segmentation
pwd
. venv/bin/activate
python train_basic.py
