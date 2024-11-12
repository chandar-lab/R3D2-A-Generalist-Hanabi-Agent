#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=96G
#SBATCH --time=95:59:00
#SBATCH -o /home/mila/n/nekoeiha/scratch/final_hanabi_checkpoint/logs/cool_job-%j.out

# Load necessary modules (if any)
module load libffi
module load OpenSSL/1.1
module load cuda/11.8

source ~/scratch/mtl_hanabi/bin/activate

# Constants
SEED=$1
PLAYER=$2
UPDATE_FREQ=$3
TOTAL_EPOCHS=4000
EPOCHS_PER_RUN=20
CHECKPOINT_DIR="/home/mila/n/nekoeiha/scratch/final_hanabi_checkpoint/R2D2-text-S"

CHECKPOINT_PATH="${CHECKPOINT_DIR}/${PLAYER}/${EPOCHS_PER_RUN}/${SEED}" #/${UPDATE_FREQ}

if [[ $PLAYER -eq 2 || $PLAYER -eq 3 || $PLAYER -eq 4 || $PLAYER -eq 5 ]]; then
    NUM_THREADS=40
else
    NUM_THREADS=10
fi

# Initialize start epoch
START_EPOCH=1

cp -r /home/mila/n/nekoeiha/MILA/mtl_paper_experiments* $SLURM_TMPDIR

cd $SLURM_TMPDIR/pyhanabi/

# Loop through until all epochs are completed
while [ $START_EPOCH -le $TOTAL_EPOCHS ]
do

    # Calculate end epoch for this run
    END_EPOCH=$(( START_EPOCH + EPOCHS_PER_RUN - 1 ))

    # Make sure we don't exceed TOTAL_EPOCHS
    if [ $END_EPOCH -gt $TOTAL_EPOCHS ]; then
        END_EPOCH=$TOTAL_EPOCHS
    fi

    # Find the latest checkpoint if it exists
    LATEST_CHECKPOINT=$(ls -t $CHECKPOINT_PATH/*.pthw 2>/dev/null | head -n 1)

    # move the directory to compute node

    # Construct the command to run the Python script
    if [ -z "$LATEST_CHECKPOINT" ]; then
        # No checkpoint found, start from the beginning
        python mtl_r2d2_main.py --config configs/iql_text.yaml --num_player $PLAYER --seed $SEED --num_thread $NUM_THREADS  --start_epoch $START_EPOCH --end_epoch $END_EPOCH --save_dir $CHECKPOINT_PATH --num_lm_layer 1 --update_freq_text_enc $UPDATE_FREQ # --lm_weights "random"
        kill -9 %
        kill -9 %
        kill -9 %
        kill -9 %
    else
        # Start from the latest checkpoint
        START_EPOCH=$(echo "$LATEST_CHECKPOINT" | grep -oP 'epoch\K[0-9]+')
        if [ -z "$START_EPOCH" ]; then
            echo "Error: Failed to extract start epoch from checkpoint."
            exit 1
        fi
        END_EPOCH=$(( START_EPOCH + EPOCHS_PER_RUN - 1 ))

        kill -9 %
        kill -9 %
        kill -9 %
        kill -9 %
        python mtl_r2d2_main.py --config configs/iql_text.yaml --num_player $PLAYER --seed $SEED --num_thread $NUM_THREADS --start_epoch $START_EPOCH --end_epoch $END_EPOCH --load_model "$LATEST_CHECKPOINT" --save_dir $CHECKPOINT_PATH  --num_lm_layer 1  --update_freq_text_enc $UPDATE_FREQ #  --lm_weights "random"
        kill -9 %
        kill -9 %
        kill -9 %
        kill -9 %
    fi

    # Update start epoch for the next iteration
    START_EPOCH=$(( END_EPOCH + 1 ))

done
scancel $SLURM_JOB_ID
