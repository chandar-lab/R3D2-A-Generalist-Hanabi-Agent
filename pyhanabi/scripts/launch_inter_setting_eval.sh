#!/bin/bash

# Configuration
NUM_PLAYERS=3
M_PLAYER_COUNT=5
SCRATCH=~/scratch/r3d2_hanabi
INCLUDE_R2D2_TEXT=false
INCLUDE_RANDOM=false  # Set to false to exclude random agents

echo "Running evaluation for $NUM_PLAYERS-player games with $M_PLAYER_COUNT-player trained agents"
echo "Include r2d2_text agents: $INCLUDE_R2D2_TEXT"
echo "Seeds for m-player agents: ${SEEDS[*]}"
echo "Include random agents: $INCLUDE_RANDOM"

# Define n-player seed variants based on NUM_PLAYERS
if [ $NUM_PLAYERS -eq 2 ]; then
    N_PLAYER_SEEDS=(a)
    SEEDS=(a)
elif [ $NUM_PLAYERS -eq 3 ]; then
    N_PLAYER_SEEDS=(a b)
    SEEDS=(a b)
elif [ $NUM_PLAYERS -eq 4 ]; then
    N_PLAYER_SEEDS=(a b c)
    SEEDS=(a b c)
elif [ $NUM_PLAYERS -eq 5 ]; then
    N_PLAYER_SEEDS=(a b c d)
    SEEDS=(a b c d)
else
    echo "Error: NUM_PLAYERS must be 2, 3, 4, or 5"
    exit 1
fi

echo "N-player seed variants: ${N_PLAYER_SEEDS[*]}"

# Define n-player trained agents (trained on NUM_PLAYERS-player games)
n_player_agents=()
for variant in "${N_PLAYER_SEEDS[@]}"; do
    n_player_agents+=("${SCRATCH}/r3d2/${NUM_PLAYERS}/20/${variant}/epoch2000.pthw")
    if [ "$INCLUDE_R2D2_TEXT" = true ]; then
        n_player_agents+=("${SCRATCH}/r2d2_text/${NUM_PLAYERS}/20/${variant}/epoch2000.pthw")
    fi
done

# Shuffle the n-player agents to ensure equal distribution
# This ensures each agent appears in roughly the same number of combinations
for i in $(seq $((${#n_player_agents[@]} - 1)) -1 1); do
    j=$((RANDOM % (i + 1)))
    temp="${n_player_agents[$i]}"
    n_player_agents[$i]="${n_player_agents[$j]}"
    n_player_agents[$j]="$temp"
done

echo "Shuffled n-player agents:"
for i in "${!n_player_agents[@]}"; do
    echo "  [$i] ${n_player_agents[$i]}"
done

# Define m-player trained agents (only M_PLAYER_COUNT-player agents)
m_player_agents=()
for variant in "${SEEDS[@]}"; do
    m_player_agents+=("${SCRATCH}/r3d2/${M_PLAYER_COUNT}/20/${variant}/epoch2000.pthw")
    if [ "$INCLUDE_R2D2_TEXT" = true ]; then
        m_player_agents+=("${SCRATCH}/r2d2_text/${M_PLAYER_COUNT}/20/${variant}/epoch2000.pthw")
    fi
done

# Add random agents (if enabled)
if [ "$INCLUDE_RANDOM" = true ]; then
    m_player_agents+=("${SCRATCH}/random_agent/2p/epoch2000.pthw")
fi

echo "Number of n-player agents: ${#n_player_agents[@]}"
echo "Number of m-player agents: ${#m_player_agents[@]}"

# Function to check if a combination has duplicate models
has_duplicates() {
    local combination="$1"
    IFS=',' read -ra models <<< "$combination"
    local seen=()
    
    for model in "${models[@]}"; do
        # Extract the full model identifier (model type, player count, and variant)
        # Remove only the epoch part
        model_id="${model%/*epoch*}"
        
        # Check if we've seen this exact model before
        for seen_model in "${seen[@]}"; do
            if [ "$model_id" = "$seen_model" ]; then
                return 0  # Has duplicates
            fi
        done
        
        # Add to seen list
        seen+=("$model_id")
    done
    
    return 1  # No duplicates
}

# Function to check if a combination has the correct number of models
has_correct_count() {
    local combination="$1"
    IFS=',' read -ra models <<< "$combination"
    local count=0
    
    for model in "${models[@]}"; do
        if [ -n "$model" ]; then  # Check if model is not empty
            count=$((count + 1))
        fi
    done
    
    [ $count -eq $NUM_PLAYERS ]
}

# Function to generate all combinations of k elements from an array
# Usage: generate_combinations array_name k
# Returns combinations in global array 'combinations'
generate_combinations() {
    local array_name="$1"
    local k="$2"
    combinations=()
    
    if [ $k -eq 0 ]; then
        combinations+=("")
        return
    fi
    
    # Get the array elements using eval (compatible with older bash)
    local temp_array=()
    eval "temp_array=(\"\${${array_name}[@]}\")"
    
    if [ $k -eq 1 ]; then
        for element in "${temp_array[@]}"; do
            combinations+=("$element")
        done
        return
    fi
    
    # Generate combinations recursively
    local n=${#temp_array[@]}
    if [ $k -gt $n ]; then
        return
    fi
    
    # Use a recursive approach to generate combinations
    local temp_combinations=()
    
    # Helper function to generate combinations recursively
    generate_combinations_recursive() {
        local start="$1"
        local current_k="$2"
        local current_combination="$3"
        
        if [ $current_k -eq 0 ]; then
            temp_combinations+=("$current_combination")
            return
        fi
        
        for ((i=start; i<=n-current_k; i++)); do
            if [ -n "$current_combination" ]; then
                generate_combinations_recursive $((i+1)) $((current_k-1)) "${current_combination},${temp_array[i]}"
            else
                generate_combinations_recursive $((i+1)) $((current_k-1)) "${temp_array[i]}"
            fi
        done
    }
    
    generate_combinations_recursive 0 $k ""
    combinations=("${temp_combinations[@]}")
}

# Generate all combinations for different subsets i < n
all_combinations=()

# For each subset size i from 1 to n-1
for subset_size in $(seq 1 $((NUM_PLAYERS-1))); do
    echo "Generating combinations for subset size: $subset_size"
    
    # Check if we have enough n-player agents for this subset size
    if [ ${#n_player_agents[@]} -lt $subset_size ]; then
        echo "Skipping subset size $subset_size: not enough n-player agents (need at least $subset_size, have ${#n_player_agents[@]})"
        continue
    fi
    
    # Generate all combinations of n-player agents of size subset_size
    generate_combinations n_player_agents $subset_size
    n_combinations=("${combinations[@]}")
    
    # For each combination of n-player agents, fill remaining slots with m-player agents
    remaining_slots=$((NUM_PLAYERS - subset_size))
    
    if [ ${#m_player_agents[@]} -lt $remaining_slots ]; then
        echo "Skipping subset size $subset_size: not enough m-player agents (need at least $remaining_slots, have ${#m_player_agents[@]})"
        continue
    fi
    
    # Generate all combinations of m-player agents of size remaining_slots
    generate_combinations m_player_agents $remaining_slots
    m_combinations=("${combinations[@]}")
    
    # Combine n-player and m-player combinations
    for n_combination in "${n_combinations[@]}"; do
        for m_combination in "${m_combinations[@]}"; do
            if [ -n "$n_combination" ] && [ -n "$m_combination" ]; then
                combination="${n_combination},${m_combination}"
            elif [ -n "$n_combination" ]; then
                combination="$n_combination"
            elif [ -n "$m_combination" ]; then
                combination="$m_combination"
            else
                continue
            fi
            
            if ! has_duplicates "$combination" && has_correct_count "$combination"; then
                all_combinations+=("$combination")
            fi
        done
    done
done

# Sort all combinations to ensure consistent ordering
IFS=$'\n' all_combinations=($(sort <<<"${all_combinations[*]}"))
unset IFS

echo "Total number of combinations: ${#all_combinations[@]}"

# Print first 10 combinations for verification
echo "First 10 combinations:"
for i in $(seq 0 $((${#all_combinations[@]} - 1))); do
    if [ $i -lt ${#all_combinations[@]} ]; then
        echo "${all_combinations[$i]}"
        echo "--------------------------------"
    fi
done

# # Run evaluations
# for combination in "${all_combinations[@]}"; do
#     echo "Running evaluation with --num_player $NUM_PLAYERS --weights $combination"
    
#     # Retry logic: try to run the Python script up to 3 times if it fails
#     max_attempts=3
#     attempt=1
#     success=0
    
#     while [ $attempt -le $max_attempts ]; do
#         python eval_model_diff.py --weights "$combination" --num_player $NUM_PLAYERS
#         if [ $? -eq 0 ]; then
#             echo "Evaluation succeeded on attempt $attempt."
#             success=1
#             break
#         else
#             echo "Evaluation failed on attempt $attempt."
#             attempt=$((attempt + 1))
#         fi
#     done
    
#     if [ $success -ne 1 ]; then
#         echo "Evaluation failed after $max_attempts attempts. Skipping..."
#     else
#         echo "After the eval"
#     fi
# done

echo "All eval done processed."
scancel $SLURM_JOB_ID