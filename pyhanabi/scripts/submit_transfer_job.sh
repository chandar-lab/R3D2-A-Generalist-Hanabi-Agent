#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=48G
#SBATCH --time=48:59:00
#SBATCH -o ${SCRATCH}/llm_hanabi_hive/slurm/cool_job-%j.out

LOAD_MODEL=$1
SEED=$2
LM_WEIGHT=$3
PLAYER=$4
ADDLAYER=$5
WANDB="1"

source ~scratch/r3d3_hanabi/bin/activate
module load libffi
module load cuda/11.8
module load OpenSSL/1.1

python r2d2_main.py --config configs/r3d2.yaml --load_model $1 --seed $SEED --num_player $PLAYER --wandb $WANDB --num_of_additional_layer $ADDLAYER