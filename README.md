# Goal
Traditional MARL systems struggle to adapt to new settings or unfamiliar collaborators, as seen in the Hanabi benchmark. This paper introduces a generalist Hanabi agent that uses language-based reformulation and a distributed MARL algorithm to enable adaptability. The proposed agent plays across all settings, transfers strategies, and collaborates with diverse agents, overcoming existing limitations.


![R3D2 Architecture](r3d2_archi.png)

This code base is based on  [Language Instructed Reinforcement Learning for Human-AI Coordination (ICML 2023)](https://github.com/hengyuan-hu/instruct-rl).
The code has been tested with PyTorch 2.0.1

## Get Started

Clone the repo with `--recursive` to include submodules
```bash
git clone --recursive git@github.com:user_name/generalist_hanabi_agent.git
```

Dependencies
```bash
pip install tdqm scipy matplotlib 'transformers[torch]'
pip install openai
```

## Table of Contents

- [Environment Setup](#environment-setup)
- [Dependencies](#dependencies)
- [GPU Configuration](#gpu-configuration)
- [Building Tokenizers](#building-tokenizers)
- [Additional Information](#additional-information)
- [Training Scripts R2D2, R3D2](#batch-job-submission-guide)
- [Evaluation Scripts Job](evaluation-job-scrips)


---


## Hanabi Learning Environment Setup

This repository contains the setup instructions for the Hanabi learning environment and related dependencies.


## Environment Setup

1. **Create a Conda environment**:
   ```bash
   conda create --name hanabi_instruct python=3.9
   ```
2. **Activate the environment**:
   ```bash
   conda activate hanabi_instruct
   ```

## Dependencies

1. **Install PyTorch (CUDA 11.8)**:
   ```bash
   pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Install Transformers library**:
   ```bash
   pip install transformers==4.31.0
   ```

3. **Load Python module**:
   ```bash
   module load python/3.9
   ```

4. **Install additional Python packages**:
   ```bash
   pip install cmake tabulate cffi psutil
   pip install tdqm scipy matplotlib wandb
   ```

## GPU Configuration

1. **Load CUDA module**:
   ```bash
   module load cuda/11.8
   ```

2. **Install Rust for building tokenizers**:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

## Building Tokenizers

1. **Navigate to the `hanabi_lib` directory**:
   ```bash
   cd hanabi-learning-environment/hanabi_lib/
   ```

2. **Clone the `tokenizers-cpp` repository**:
   ```bash
   git clone --recursive git@github.com:mlc-ai/tokenizers-cpp.git
   ```

## Build
   ```bash
   make
   ```
---

## Additional Information

- Ensure that all required modules are properly loaded in your environment.
- The setup assumes access to a GPU with CUDA 11.8 support.
- Use `wandb` for experiment tracking and logging during model training.
- If issues arise during the setup, check for compatibility between package versions and your system configuration.

Feel free to contribute or report issues!


## Training Scripts R2D2, R3D2

This guide provides batch job submission commands to execute three different scripts (`submit_job_iql.sh`, `submit_jobs_other_player.sh`, and `launch_r3d2.sh`) for multiple models and players.

## Commands
```bash
cd pyhanabi
```

### Submitting Jobs for `submit_job_iql.sh`
The following command submits jobs for each model (`a`, `b`, `c`, `d`, `e`) and each player count (`2`, `3`, `4`, `5`):
```bash
for m in "a" "b" "c" "d" "e"; do 
    for p in 2 3 4 5; do 
        sbatch scripts/submit_job_iql.sh $m $p; 
    done; 
done;
```

### Submitting Jobs for `submit_jobs_other_player.sh`
The following command submits jobs for `submit_jobs_other_player.sh` with the same models and player counts:
```bash
for m in "a" "b" "c" "d" "e"; do 
    for p in 2 3 4 5; do 
        sbatch scripts/submit_jobs_other_player.sh $m $p; 
    done; 
done;
```

### Submitting Jobs for Single-task R3D2 `launch_r3d2.sh`
The following command submits jobs for `launch_r3d2.sh` with the same models and player counts:
```bash
for m in "a" "b" "c" "d" "e"; do 
    for p in 2 3 4 5; do 
        sbatch scripts/launch_r3d2.sh $m $p; 
    done; 
done;
```

###  Submitting Jobs for Multi-task R3D2 `launch_r3d2.sh` (Player 6)
The following command submits jobs for `launch_r3d2.sh` for `Player 6`, representing multi-task R3D2, with the same models:

```bash

for m in "a" "b" "c" "d" "e"; do 
    for p in 6; do 
        sbatch scripts/launch_r3d2.sh $m $p; 
    done; 
done;
```

## Explanation

- **Seed (`m`)**: The scripts will iterate over the models `a`, `b`, `c`, `d`, and `e`.
- **Player Setting (`p`)**: Jobs will be submitted for player counts of `2`, `3`, `4`, and `5`.
- **Scripts**: The three scripts handle different types of job submissions (`IQL`, `other players`, and `R3D2`).



## Evaluation Scripts Job

```bash
scripts/launch_2p_eval_diff_setting_all.sh
scripts/launch_3p_eval_diff_setting_all.sh
scripts/launch_4p_eval_diff_setting_all.sh
scripts/launch_5p_eval_diff_setting_all.sh

scripts/launch_cross_play.sh


```

