#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=48G
#SBATCH --time=7:59:00
#SBATCH -o ${SCRATCH}/final_hanabi_checkpoint/eval_logs/cool_job-%j.out


# Load necessary modules (if any)
module load libffi
module load OpenSSL/1.1
module load cuda/11.8   # Example: adjust to your environment
source ~/scratch/r3d3_hanabi/bin/activate

SEED=$1

CHECKPOINT_DIR=$2

players=('${SCRATCH}/v2_hanabi_checkpoints_r3d2/2/20/c/epoch2500.pthw') # 'IQL-5a'

echo "Sorting files by epoch number:"
alternatives_2=$(ls $CHECKPOINT_DIR/*.pthw 2>/dev/null | \
    sed 's/.*epoch\([0-9]\+\).pthw/\1 &/' | \
    sort -n | \
    cut -d' ' -f2-)
echo "alternatives_2 ${alternatives_2}"

pair_list=()
for ((i=0; i<${#players[@]}; i++)); do
    for file in $alternatives_2; do
        pair_list+=("${players[$i]},${file}")
    done
done

# Print the combinations and total
printf '%s\n' "${pair_list[@]}"
echo "Total number of combinations: ${#pair_list[@]}"



# echo "Generated pairs: ${pair_list[@]}"
for pair in "${pair_list[@]}"; do
      echo "Running evaluation for $file with --num_player 2 --weights $pair --seed $SEED"

      # Retry logic: try to run the Python script up to 3 times if it fails
      max_attempts=3
      attempt=1
      success=0

      while [ $attempt -le $max_attempts ]; do
          python eval_model_diff.py --weights "$pair" --num_player 2 --seed $SEED
          if [ $? -eq 0 ]; then
              echo "Evaluation succeeded on attempt $attempt."
              success=1
              break
          else
              echo "Evaluation failed on attempt $attempt."
              attempt=$((attempt + 1))
          fi
      done

      if [ $success -ne 1 ]; then
          echo "Evaluation failed after $max_attempts attempts. Skipping..."
      else
          echo "After the eval"
      fi
done



echo "All eval done processed."
scancel $SLURM_JOB_ID