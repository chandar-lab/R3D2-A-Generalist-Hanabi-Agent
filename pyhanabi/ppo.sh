#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:59:00
#SBATCH -o /home/mila/a/arjun.vaithilingam-sudhakar/scratch/Zeroshot_hanabi_instructrl/pyhanabi/baseline_logs/ppo-%j.out

llm_prior_knowledge=$1
seed=$2

module load libffi
module load cuda/11.2

python ppo_main.py \
   --num_thread 40 \
   --seed $seed \
  --num_game_per_thread 120 \
  --replay_buffer_size 1024 \
  --train_device cuda:0 \
  --act_device cuda:0 \
  --epoch_len 5000 \
  --gamma 0.999 \
  --net publ-lstm \
  --shuffle_color 0 \
  --target_data_ratio 2 \
  --llm_prior $llm_prior_knowledge \
  --num_epoch 150 \
  --pikl_beta 2 \
  --pikl_lambda 0.05 \
  --pikl_anneal_epoch 50 \
  --save_per 50

