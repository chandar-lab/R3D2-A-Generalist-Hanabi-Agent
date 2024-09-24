#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=48G
#SBATCH --time=71:59:00
#SBATCH -o /home/mila/m/mathieu.reymond/scratch/v2_hanabi_logs/cool_job-%j.out

# Load necessary modules (if any)
module load libffi
module load OpenSSL/1.1
module load cuda/11.8   # Example: adjust to your environment

# Constants
SEED=$1
PLAYER=$2
TOTAL_EPOCHS=2500
EPOCHS_PER_RUN=20
CHECKPOINT_DIR="/home/mila/m/mathieu.reymond/scratch/v2_hanabi_checkpoints_r3d2"

CHECKPOINT_PATH="${CHECKPOINT_DIR}/${PLAYER}/${EPOCHS_PER_RUN}/${SEED}"

if [[ $PLAYER -eq 2 || $PLAYER -eq 3 || $PLAYER -eq 4 || $PLAYER -eq 5 ]]; then
    NUM_THREADS=40
else
    NUM_THREADS=10
fi

# Initialize start epoch
START_EPOCH=1

cp -r /home/mila/m/mathieu.reymond/dev/v2_hanabi_multi_task/Zeroshot_hanabi_instructrl/* $SLURM_TMPDIR

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
        python mtl_r2d2_main.py --config configs/drrn_mtl.yaml --num_player $PLAYER --seed $SEED --num_thread $NUM_THREADS  --start_epoch $START_EPOCH --end_epoch $END_EPOCH --save_dir $CHECKPOINT_PATH
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
        python mtl_r2d2_main.py --config configs/drrn_mtl.yaml --num_player $PLAYER --seed $SEED --num_thread $NUM_THREADS --start_epoch $START_EPOCH --end_epoch $END_EPOCH --load_model "$LATEST_CHECKPOINT" --save_dir $CHECKPOINT_PATH
        kill -9 %
        kill -9 %
        kill -9 %
        kill -9 %
    fi

    # Update start epoch for the next iteration
    START_EPOCH=$(( END_EPOCH + 1 ))

done
scancel $SLURM_JOB_ID
