#!/bin/bash

BASE_PATH="/home/mila/m/mathieu.reymond/scratch/v2_hanabi_checkpoints_r3d2"
numbers=(2 3 4 5)
letters=(a b c d e)
seeds=(1111 2222 3333 4444 5555)

for num in "${numbers[@]}"; do
    for i in {0..4}; do
        letter=${letters[$i]}
        seed=${seeds[$i]}
        sbatch launch_eval_model.sh "$BASE_PATH/$num/20/$letter/" $num "r3d2" $seed
    done
done

BASE_PATH="/home/mila/n/nekoeiha/scratch/final_hanabi_checkpoint/R2D2-text-S/"
numbers=(2 3 4 5)
letters=(a b c)
seeds=(1111 2222 3333)

for num in "${numbers[@]}"; do
    for i in {0..2}; do
        letter=${letters[$i]}
        seed=${seeds[$i]}
        sbatch launch_eval_model.sh "$BASE_PATH/$num/20/$letter/" $num "r2d2_text" $seed
    done
done

