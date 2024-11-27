#!/bin/sh

#SBATCH --partition=GPUQ
#SBATCH --job-name=v5/SwinUNETR
#SBATCH --account=share-ie-idi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=16G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=(a100)
#SBATCH --output=logs/%x/idun/out-%j.log
#SBATCH --error=logs/%x/idun/out-%j.log
#SBATCH --nodelist=idun-06-[01-07]

echo $SLURM_JOB_NAME-$SLURM_JOB_ID
cd /cluster/work/andrebw/medical-image-segmentation
pwd
. venv/bin/activate

nvidia-smi

python run_training.py \
  --experiment-id $SLURM_JOB_NAME  \
  --model-type SwinUNETR \
  --batch-size 4  \
  --epochs 700  \
  --lr-init 1e-4  \
  --lr-min 1e-15  \
  --weight-decay 1e-5  \
  --smooth-nr 0  \
  --smooth-dr 1e-5  \
  --cpu-cores $SLURM_CPUS_ON_NODE \
#  --no-training
#  --checkpoint "logs/v2/HNTS-UNETR/checkpoints/16500" \
