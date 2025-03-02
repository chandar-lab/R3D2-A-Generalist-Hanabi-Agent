#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=48G
#SBATCH --time=71:59:00
#SBATCH -o /home/mila/n/nekoeiha/scratch/llm_hanabi_hive/slurm/cool_job-%j.out


FREQ=$1
SEED=$2
LM_WEIGHT=$3
PLAYER=$4
ADDLAYER=$5
WANDB="1"

source /home/mila/n/nekoeiha/scratch/llm_instruct_rl/bin/activate
module load libffi
module load cuda/11.8
module load OpenSSL/1.1

python r2d2_main.py --config configs/r3d2.yaml --update_freq_text_enc $FREQ --seed $SEED --lm_weights $LM_WEIGHT --num_player $PLAYER --wandb $WANDB --num_of_additional_layer $ADDLAYER