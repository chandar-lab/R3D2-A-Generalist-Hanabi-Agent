#!/bin/bash

# Configuration
NUM_PLAYERS=3
M_PLAYER_COUNT=4
SCRATCH=~/scratch/r3d2_hanabi
INCLUDE_R2D2_TEXT=false
INCLUDE_RANDOM=false  # Set to false to exclude random agents
SEEDS=(a b c)  # List of seeds for m-player agents

echo "Running evaluation for $NUM_PLAYERS-player games with $M_PLAYER_COUNT-player trained agents"
echo "Include r2d2_text agents: $INCLUDE_R2D2_TEXT"
echo "Seeds for m-player agents: ${SEEDS[*]}"
echo "Include random agents: $INCLUDE_RANDOM"
# Define n-player trained agents (trained on NUM_PLAYERS-player games)
n_player_agents=()
for variant in a b; do
    n_player_agents+=("${SCRATCH}/r3d2/${NUM_PLAYERS}/20/${variant}/epoch2000.pthw")
    if [ "$INCLUDE_R2D2_TEXT" = true ]; then
        n_player_agents+=("${SCRATCH}/r2d2_text/${NUM_PLAYERS}/20/${variant}/epoch2000.pthw")
    fi
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

# Generate all combinations for different subsets i < n
all_combinations=()

# For each subset size i from 1 to n-1
for subset_size in $(seq 1 $((NUM_PLAYERS-1))); do
    echo "Generating combinations for subset size: $subset_size"
    
    # Generate all combinations of n-player agents of size subset_size
    if [ $subset_size -eq 1 ]; then
        # Single n-player agent combinations
        for n_agent in "${n_player_agents[@]}"; do
            # Fill remaining slots with m-player agents
            remaining_slots=$((NUM_PLAYERS - subset_size))
            
            if [ $remaining_slots -eq 1 ]; then
                # Single m-player agent
                for m_agent in "${m_player_agents[@]}"; do
                    combination="${n_agent},${m_agent}"
                    if ! has_duplicates "$combination" && has_correct_count "$combination"; then
                        all_combinations+=("$combination")
                    fi
                done
            elif [ $remaining_slots -eq 2 ]; then
                # Two m-player agents
                for i in $(seq 0 $((${#m_player_agents[@]}-1))); do
                    for j in $(seq $((i+1)) $((${#m_player_agents[@]}-1))); do
                        m_agent1="${m_player_agents[$i]}"
                        m_agent2="${m_player_agents[$j]}"
                        combination="${n_agent},${m_agent1},${m_agent2}"
                        if ! has_duplicates "$combination" && has_correct_count "$combination"; then
                            all_combinations+=("$combination")
                        fi
                    done
                done
            fi
        done
    elif [ $subset_size -eq 2 ]; then
        # Two n-player agent combinations - only if we have at least 2 n-player agents
        if [ ${#n_player_agents[@]} -ge 2 ]; then
            for i in $(seq 0 $((${#n_player_agents[@]}-2))); do
                for j in $(seq $((i+1)) $((${#n_player_agents[@]}-1))); do
                    n_agent1="${n_player_agents[$i]}"
                    n_agent2="${n_player_agents[$j]}"
                    
                    # Fill remaining slot with m-player agent
                    remaining_slots=$((NUM_PLAYERS - subset_size))
                    
                    if [ $remaining_slots -eq 1 ]; then
                        for m_agent in "${m_player_agents[@]}"; do
                            combination="${n_agent1},${n_agent2},${m_agent}"
                            if ! has_duplicates "$combination" && has_correct_count "$combination"; then
                                all_combinations+=("$combination")
                            fi
                        done
                    fi
                done
            done
        else
            echo "Skipping subset size 2: not enough n-player agents (need at least 2, have ${#n_player_agents[@]})"
        fi
    fi
done

echo "Total number of combinations: ${#all_combinations[@]}"

# Print first 10 combinations for verification
echo "First 10 combinations:"
for i in $(seq 0 9); do
    if [ $i -lt ${#all_combinations[@]} ]; then
        echo "${all_combinations[$i]}"
    fi
done

# Run evaluations
for combination in "${all_combinations[@]}"; do
    echo "Running evaluation with --num_player $NUM_PLAYERS --weights $combination"
    
    # Retry logic: try to run the Python script up to 3 times if it fails
    max_attempts=3
    attempt=1
    success=0
    
    while [ $attempt -le $max_attempts ]; do
        python eval_model_diff.py --weights "$combination" --num_player $NUM_PLAYERS
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