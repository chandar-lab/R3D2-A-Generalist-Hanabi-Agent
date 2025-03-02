#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=48G
#SBATCH --time=71:59:00
#SBATCH -o ${SCRATCH}/final_hanabi_checkpoint/logs/cool_job-%j.out

# Load necessary modules (if any)
module load libffi
module load OpenSSL/1.1
module load cuda/11.8

source ~/scratch/r3d3_hanabi/bin/activate

python selfplay.py \
       --save_dir exps/iql \
       --num_thread 80 \
       --num_game_per_thread 80 \
       --method iql \
       --sad 0 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --gamma 0.999 \
       --seed 2254257 \
       --burn_in_frames 10000 \
       --replay_buffer_size 100000 \
       --batchsize 128 \
       --epoch_len 1000 \
       --num_epoch 1 \
       --num_player 2 \
       --net lstm \
       --num_lstm_layer 2 \
       --multi_step 3 \
       --train_device cuda:0 \
       --act_device cuda:1 \

python train_belief.py \
       --save_dir exps/belief_obl0 \
       --num_thread 40 \
       --num_game_per_thread 40 \
       --batchsize 128 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --hid_dim 512 \
       --burn_in_frames 10000 \
       --replay_buffer_size 100000 \
       --epoch_len 1000 \
       --num_epoch 500 \
       --train_device cuda:0 \
       --act_device cuda:1 \
       --explore 1 \
       --policy exps/iql/model0.pthw \
       --seed 2254257 \
       --num_player 2 \
       --shuffle_color 0 \
       --rand 1 \

python selfplay.py \
       --save_dir exps/obl1 \
       --num_thread 80 \
       --num_game_per_thread 80 \
       --sad 0 \
       --act_base_eps 0.1 \
       --act_eps_alpha 7 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --gamma 0.999 \
       --seed 2254257 \
       --batchsize 128 \
       --burn_in_frames 10000 \
       --replay_buffer_size 100000 \
       --epoch_len 1000 \
       --num_epoch 1500 \
       --num_player 2 \
       --rnn_hid_dim 512 \
       --multi_step 1 \
       --train_device cuda:0 \
       --act_device cuda:1,cuda:2 \
       --num_lstm_layer 2 \
       --boltzmann_act 0 \
       --min_t 0.01 \
       --max_t 0.1 \
       --off_belief 1 \
       --num_fict_sample 10 \
       --belief_device cuda:3,cuda:4 \
       --belief_model exps/belief_obl0/model0.pthw \
       --load_model None \
       --net publ-lstm