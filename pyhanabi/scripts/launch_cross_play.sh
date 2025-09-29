#!/bin/bash

# Load necessary modules (if any)
module load libffi
module load OpenSSL/1.1
module load cuda/11.8

source ~/scratch/r3d2_hanabi/bin/activate

# Constants
num_player=2

CHECKPOINT_DIR="${SCRATCH}/final_hanabi_checkpoint/zero_shot_eval_2p"

pthw_files=$(find "$CHECKPOINT_DIR" -type f -name "*.pthw")

pthw_files_list=""

for file in $pthw_files; do
    pthw_files_list="${pthw_files_list}\"$file\", "
done

pthw_files_list="[${pthw_files_list%, }]"  # Surround with square brackets to make it a list

echo "List of .pthw files: $pthw_files_list"

string_list=$(python <<END
import itertools
import json
models = $pthw_files_list
num_player = $num_player
combs = list(itertools.combinations_with_replacement(models, num_player))
string_list = [f"{x[0]}+{x[1]}" for x in combs]
print(json.dumps(string_list))
END
)

echo "Combinations: $string_list"


MAX_RETRIES=3

for file in $string_list; do
    echo "Processing $file"
    kill -9 %
    kill -9 %
    retries=0
    success=0

    while [ $retries -lt $MAX_RETRIES ]; do
        python tools/cross_play.py --root $file

        # Check the exit status of the last command
        if [ $? -eq 0 ]; then
            success=1
            echo "Successfully processed $file"
            break
        else
            echo "Failed to process $file, retrying... ($((retries + 1))/$MAX_RETRIES)"
            retries=$((retries + 1))
        fi
    done

    # If script still failed after retries, print error message
    if [ $success -eq 0 ]; then
        echo "Error: Failed to process $file after $MAX_RETRIES attempts."
    fi

done
echo "All eval done processed."
scancel $SLURM_JOB_ID