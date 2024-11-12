#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=48G
#SBATCH --time=7:59:00
#SBATCH -o /home/mila/n/nekoeiha/scratch/final_hanabi_checkpoint/eval_logs/cool_job-%j.out


# Load necessary modules (if any)
module load libffi
module load OpenSSL/1.1
module load cuda/11.8   # Example: adjust to your environment
source ~/scratch/mtl_hanabi/bin/activate

SEED=$1
cp -r /home/mila/n/nekoeiha/MILA/mtl_paper_experiments_no_buffer_saving_both_vec_text/* $SLURM_TMPDIR

cd $SLURM_TMPDIR/pyhanabi/


players=('/home/mila/m/mathieu.reymond/scratch/v2_hanabi_checkpoints_r3d2/2/20/a/epoch2500.pthw' '/home/mila/n/nekoeiha/scratch/final_hanabi_checkpoint/R2D2-text-S/2/20/b/epoch2500.pthw' ) # 'IQL-5a'
alternatives_2=('/home/mila/m/mathieu.reymond/scratch/v2_hanabi_checkpoints_r3d2/2/20/b/epoch2500.pthw' '/home/mila/n/nekoeiha/scratch/final_hanabi_checkpoint/R2D2-text-S/2/20/c/epoch2500.pthw' '/home/mila/m/mathieu.reymond/scratch/v2_hanabi_checkpoints_r3d2/3/20/a/epoch2500.pthw' '/home/mila/m/mathieu.reymond/scratch/v2_hanabi_checkpoints_r3d2/4/20/a/epoch2500.pthw' '/home/mila/m/mathieu.reymond/scratch/v2_hanabi_checkpoints_r3d2/5/20/b/epoch2520.pthw' '/home/mila/n/nekoeiha/scratch/final_hanabi_checkpoint/R2D2-text-S/3/20/a/epoch3000.pthw' '/home/mila/n/nekoeiha/scratch/final_hanabi_checkpoint/R2D2-text-S/4/20/a/epoch2480.pthw' '/home/mila/n/nekoeiha/scratch/final_hanabi_checkpoint/R2D2-text-S/5/20/b/epoch2040.pthw' '/home/mila/n/nekoeiha/scratch/final_hanabi_checkpoint/multitask_learning_no_saved_buffer/6/20/a/epoch3320.pthw' '/home/mila/n/nekoeiha/scratch/final_hanabi_checkpoint/random_agent/2p/epoch0.pthw') # 'IQL-5b' 'R3-2b' 'R3-3b' 'R3-4b' 'R2T-2b' 'R2T-3b' 'R2T-4b'

pair_list=()
for ((i=0; i<${#players[@]}; i++)); do
    for ((j=0; j<${#alternatives_2[@]}; j++)); do
        if [[ $j -ge $i ]]; then
            pair_list+=("${players[$i]},${alternatives_2[$j]}")
        fi
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