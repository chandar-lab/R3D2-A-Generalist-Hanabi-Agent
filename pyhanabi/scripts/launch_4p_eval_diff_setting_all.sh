#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=48G
#SBATCH --time=22:59:00
#SBATCH -o ${SCRATCH}/final_hanabi_checkpoint/eval_logs/cool_job-%j.out

# Load necessary modules (if any)
module load libffi
module load OpenSSL/1.1
module load cuda/11.8   # Example: adjust to your environment
source ~/scratch/r3d3_hanabi/bin/activate

SEED=$1
players=(
  "${SCRATCH}/v2_hanabi_checkpoints_r3d2/4/20/a/epoch2500.pthw"
  "${SCRATCH}/final_hanabi_checkpoint/R2D2-text-S/4/20/a/epoch2480.pthw"
) # 'IQL-5a'

alternatives_2=(
  "${SCRATCH}/v2_hanabi_checkpoints_r3d2/4/20/b/epoch2500.pthw"
  "${SCRATCH}/final_hanabi_checkpoint/R2D2-text-S/4/20/b/epoch2500.pthw"
  "${SCRATCH}/final_hanabi_checkpoint/R2D2-text-S/2/20/b/epoch2000.pthw"
  "${SCRATCH}/final_hanabi_checkpoint/R2D2-text-S/3/20/b/epoch2500.pthw"
  "${SCRATCH}/final_hanabi_checkpoint/R2D2-text-S/5/20/a/epoch2000.pthw"
  "${SCRATCH}/v2_hanabi_checkpoints_r3d2/2/20/a/epoch2500.pthw"
  "${SCRATCH}/v2_hanabi_checkpoints_r3d2/3/20/a/epoch2500.pthw"
  "${SCRATCH}/v2_hanabi_checkpoints_r3d2/5/20/a/epoch2500.pthw"
  "${SCRATCH}/final_hanabi_checkpoint/multitask_learning_no_saved_buffer/6/20/a/epoch3320.pthw"
  "${SCRATCH}/final_hanabi_checkpoint/random_agent/2p/epoch0.pthw"
) # 'IQL-5b' 'R3-2b' 'R3-3b' 'R3-4b' 'R2T-2b' 'R2T-3b' 'R2T-4b'

alternatives_3=(
  "${SCRATCH}/v2_hanabi_checkpoints_r3d2/4/20/c/epoch2500.pthw"
  "${SCRATCH}/final_hanabi_checkpoint/R2D2-text-S/4/20/b/epoch2800.pthw"
  "${SCRATCH}/final_hanabi_checkpoint/R2D2-text-S/2/20/c/epoch2000.pthw"
  "${SCRATCH}/final_hanabi_checkpoint/R2D2-text-S/3/20/c/epoch3000.pthw"
  "${SCRATCH}/final_hanabi_checkpoint/R2D2-text-S/5/20/b/epoch2000.pthw"
  "${SCRATCH}/v2_hanabi_checkpoints_r3d2/2/20/b/epoch2500.pthw"
  "${SCRATCH}/v2_hanabi_checkpoints_r3d2/3/20/b/epoch2500.pthw"
  "${SCRATCH}/v2_hanabi_checkpoints_r3d2/5/20/b/epoch2500.pthw"
  "${SCRATCH}/final_hanabi_checkpoint/multitask_learning_no_saved_buffer/6/20/b/epoch3280.pthw"
  "${SCRATCH}/final_hanabi_checkpoint/random_agent/2p/epoch0.pthw"
) # 'R3-2c' 'R3-3c' 'R3-4c' 'R2T-2c' 'R2T-3c' 'R2T-4c'

alternatives_4=(
  "${SCRATCH}/v2_hanabi_checkpoints_r3d2/4/20/c/epoch2560.pthw"
  "${SCRATCH}/final_hanabi_checkpoint/R2D2-text-S/4/20/c/epoch2460.pthw"
  "${SCRATCH}/final_hanabi_checkpoint/R2D2-text-S/2/20/c/epoch2000.pthw"
  "${SCRATCH}/final_hanabi_checkpoint/R2D2-text-S/3/20/c/epoch3000.pthw"
  "${SCRATCH}/final_hanabi_checkpoint/R2D2-text-S/5/20/c/epoch2000.pthw"
  "${SCRATCH}/v2_hanabi_checkpoints_r3d2/2/20/c/epoch2500.pthw"
  "${SCRATCH}/v2_hanabi_checkpoints_r3d2/3/20/c/epoch2500.pthw"
  "${SCRATCH}/v2_hanabi_checkpoints_r3d2/5/20/c/epoch2500.pthw"
  "${SCRATCH}/final_hanabi_checkpoint/multitask_learning_no_saved_buffer/6/20/b/epoch3280.pthw"
  "${SCRATCH}/final_hanabi_checkpoint/random_agent/2p/epoch0.pthw"
) # 'R3-2c' 'R3-3c' 'R3-4c' 'R2T-2c' 'R2T-3c' 'R2T-4c'
pair_list=()

for ((i=0; i<${#players[@]}; i++)); do
    for ((j=0; j<${#alternatives_2[@]}; j++)); do
        for ((k=0; k<${#alternatives_3[@]}; k++)); do
            for ((m=0; m<${#alternatives_4[@]}; m++)); do
                # Count how many R3-2* and R3-3* are in the combination
                r3_2_count=0
                r3_3_count=0
                r3_5_count=0
                r2_2_count=0
                r2_3_count=0
                r2_5_count=0

                # Check each alternative for R3-2* and R3-3*
                # Check each alternative for R3-2* and R3-3*
                [[ ${alternatives_2[$j]} == *"r3d2/2/"* ]] && ((r3_2_count++))
                [[ ${alternatives_2[$j]} == *"r3d2/3/"* ]] && ((r3_3_count++))
                [[ ${alternatives_2[$j]} == *"r3d2/5/"* ]] && ((r3_5_count++))
                [[ ${alternatives_3[$k]} == *"r3d2/2/"* ]] && ((r3_2_count++))
                [[ ${alternatives_3[$k]} == *"r3d2/3/"* ]] && ((r3_3_count++))
                [[ ${alternatives_3[$k]} == *"r3d2/5/"* ]] && ((r3_5_count++))
                [[ ${alternatives_4[$m]} == *"r3d2/2/"* ]] && ((r3_2_count++))
                [[ ${alternatives_4[$m]} == *"r3d2/3/"* ]] && ((r3_3_count++))
                [[ ${alternatives_4[$m]} == *"r3d2/5/"* ]] && ((r3_5_count++))

                [[ ${alternatives_2[$j]} == *"R2D2-text-S/2/"* ]] && ((r2_2_count++))
                [[ ${alternatives_2[$j]} == *"R2D2-text-S/3/"* ]] && ((r2_3_count++))
                [[ ${alternatives_2[$j]} == *"R2D2-text-S/5/"* ]] && ((r2_5_count++))
                [[ ${alternatives_3[$k]} == *"R2D2-text-S/2/"* ]] && ((r2_2_count++))
                [[ ${alternatives_3[$k]} == *"R2D2-text-S/3/"* ]] && ((r2_3_count++))
                [[ ${alternatives_3[$k]} == *"R2D2-text-S/5/"* ]] && ((r2_5_count++))
                [[ ${alternatives_4[$m]} == *"R2D2-text-S/2/"* ]] && ((r2_2_count++))
                [[ ${alternatives_4[$m]} == *"R2D2-text-S/3/"* ]] && ((r2_3_count++))
                [[ ${alternatives_4[$m]} == *"R2D2-text-S/5/"* ]] && ((r2_5_count++))

                # Only add combination if we don't have both R3-2* and R3-3* present
                total_positive=$(( (r3_2_count > 0) + (r2_2_count > 0) + (r3_3_count > 0) + (r2_3_count > 0) + (r3_5_count > 0) + (r2_5_count > 0) ))
                if [[ $total_positive -lt 2 ]]; then
                  if [[ $j -ge $i && $k -ge $j && $m -ge $k ]]; then
                        pair_list+=("${players[$i]},${alternatives_2[$j]},${alternatives_3[$k]},${alternatives_4[$m]}")
                    fi
                fi
            done
        done
    done
done

# Print the combinations and total
printf '%s\n' "${pair_list[@]}"
echo "Total number of combinations: ${#pair_list[@]}"



# echo "Generated pairs: ${pair_list[@]}"
for pair in "${pair_list[@]}"; do
      echo "Running evaluation for $file with --num_player 4 --weights $pair --seed $SEED"

      # Retry logic: try to run the Python script up to 3 times if it fails
      max_attempts=3
      attempt=1
      success=0

      while [ $attempt -le $max_attempts ]; do
          python eval_model_diff.py --weights "$pair" --num_player 4 --seed $SEED
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