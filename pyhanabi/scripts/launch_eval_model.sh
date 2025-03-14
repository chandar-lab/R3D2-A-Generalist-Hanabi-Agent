#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=7:59:00
#SBATCH -o ${SCRATCH}/final_hanabi_checkpoint/eval_logs/cool_job-%j.out

# Load necessary modules (if any)
module load libffi
module load OpenSSL/1.1
module load cuda/11.8

source ~/scratch/r3d3_hanabi/bin/activate
# Constants
CHECKPOINT_DIR=$1 # Accept multiple directories as arguments


echo "Processing directory: $CHECKPOINT_DIR"

echo "Sorting files by epoch number:"
CHECKPOINT_FILES=$(ls $CHECKPOINT_DIR/*.pthw 2>/dev/null | \
    sed 's/.*epoch\([0-9]\+\).pthw/\1 &/' | \
    sort -n | \
    cut -d' ' -f2-)

echo "Sorted Checkpoint Files:"
echo $CHECKPOINT_FILES

for file in $CHECKPOINT_FILES; do
    echo "Processing $file"
    python eval_model.py --weight1 $file --num_player $2 --method $3 --seed $4
done

echo "All evaluations processed."
scancel $SLURM_JOB_ID