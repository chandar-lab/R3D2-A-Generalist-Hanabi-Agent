#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=48G
#SBATCH --time=2:59:00
#SBATCH -o /home/mila/n/nekoeiha/scratch/final_hanabi_checkpoint/eval_logs/cool_job-%j.out

# Load necessary modules (if any)
module load libffi
module load OpenSSL/1.1
module load cuda/11.8   # Example: adjust to your environment
source ~/scratch/mtl_hanabi/bin/activate

SEED=$1
cp -r /home/mila/n/nekoeiha/MILA/mtl_paper_experiments_no_buffer_saving_both_vec_text/* $SLURM_TMPDIR

cd $SLURM_TMPDIR/pyhanabi/

players=('/home/mila/m/mathieu.reymond/scratch/v2_hanabi_checkpoints_r3d2/2/20/a/epoch2000.pthw' '/home/mila/m/mathieu.reymond/scratch/v2_hanabi_checkpoints_r3d2/3/20/a/epoch2000.pthw' '/home/mila/m/mathieu.reymond/scratch/v2_hanabi_checkpoints_r3d2/4/20/a/epoch2000.pthw' '/home/mila/m/mathieu.reymond/scratch/v2_hanabi_checkpoints_r3d2/5/20/a/epoch2000.pthw')
#alternatives=('/home/mila/m/mathieu.reymond/scratch/v2_hanabi_checkpoints_r3d2/2/20/b/epoch2000.pthw' '/home/mila/m/mathieu.reymond/scratch/v2_hanabi_checkpoints_r3d2/3/20/b/epoch2000.pthw' '/home/mila/m/mathieu.reymond/scratch/v2_hanabi_checkpoints_r3d2/4/20/b/epoch2000.pthw' '/home/mila/m/mathieu.reymond/scratch/v2_hanabi_checkpoints_r3d2/5/20/b/epoch2000.pthw' '/home/mila/n/nekoeiha/scratch/final_hanabi_checkpoint/R2D2-text-S/2/20/b/epoch2020.pthw' '/home/mila/n/nekoeiha/scratch/final_hanabi_checkpoint/R2D2-text-S/3/20/a/epoch2020.pthw' '/home/mila/n/nekoeiha/scratch/final_hanabi_checkpoint/R2D2-text-S/4/20/a/epoch2020.pthw' '/home/mila/n/nekoeiha/scratch/final_hanabi_checkpoint/R2D2-text-S/5/20/a/epoch2020.pthw'  '/home/mila/n/nekoeiha/scratch/final_hanabi_checkpoint/multitask_learning_no_saved_buffer/6/20/a/epoch2900.pthw'  '/home/mila/n/nekoeiha/scratch/final_hanabi_checkpoint/random_agent/2p/epoch0.pthw')
alternatives=('/home/mila/n/nekoeiha/scratch/final_hanabi_checkpoint/R2D2-text-S/4/20/b/epoch2000.pthw' '/home/mila/n/nekoeiha/scratch/final_hanabi_checkpoint/R2D2-text-S/5/20/a/epoch1480.pthw'  )

pair_list=()
counter=2  # Initialize counter from 2

for p1 in "${players[@]}"; do
    for p2 in "${alternatives[@]}"; do
        if [ "$p1" == "$p2" ]; then
            index=$(( $(echo ${players[@]} | tr ' ' '\n' | grep -n "$p1" | cut -d: -f1) - 1 ))
            p2=${alternatives[$index]}
        fi
        pair_list+=("$p1,$p2,$counter")
	# echo "p1: $p1, p2: $p2"  # Print p1 and p2

    done
    ((counter++))
done



# echo "Generated pairs: ${pair_list[@]}"
for pair in "${pair_list[@]}"; do
    IFS=',' read weight1 weight2 player <<< "$pair"
    for ((j=1; j<player; j++)); do
        echo "Running evaluation for $file with --num_player $player --weight1 $weight1 --weight2 $weight2 --num_alternative_player $j --seed $SEED"

        # Retry logic: try to run the Python script up to 3 times if it fails
        max_attempts=3
        attempt=1
        success=0

        while [ $attempt -le $max_attempts ]; do
            python eval_model_diff.py --weight1 "$weight1" --weight2 "$weight2" --num_player "$player" --num_alternative_player $j --seed $SEED
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
done



echo "All eval done processed."
scancel $SLURM_JOB_ID