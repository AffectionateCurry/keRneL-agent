# GRPO Training for Kernel Optimization

<img width="703" alt="image" src="https://github.com/user-attachments/assets/32793dee-c889-4f8d-be90-1999865d4fcc" />


## Overview

GRPO is applied to train a Qwen model to generate optimization suggestions for CUDA kernels. A blackbox LLM (GPT-4o) implements these suggestions, with the resulting kernels evaluated for compilation, correctness, and performance.

## System Architecture

```
┌─────────────────┐    Suggestion    ┌──────────────────┐    Implementation    ┌──────────────────┐
│                 │ ───────────────> │                  │ ──────────────────>  │                  │
│   Qwen Agent    │                  │  Blackbox LLM    │                      │  Kernel Compiler │
│   (Training)    │ <─────────────── ┤   (GPT-4o)       │ <──────────────────  │  & Evaluator     │
│                 │      Reward      │                  │     Performance      │                  │
└─────────────────┘                  └──────────────────┘       Metrics        └──────────────────┘
```

## Initialization Process

The training script (`train_grpo.py`) performs the following initialization steps:

1. **Load Models & Tokenizer**
   - Load Qwen model (e.g., `Qwen/Qwen2-0.5B-Instruct`) with its tokenizer
   - Create a frozen reference copy of the initial Qwen model

2. **Initialize Environment**
   - Create `KernelBenchGRPOEnv` wrapper (which initializes `KernelBenchRLEnv`)
   - Load the specified kernel benchmark dataset
   - Set up device and GPU architecture
   - Configure `gpt4o_code_generator` as the blackbox LLM

3. **Configure GRPO Trainer**
   - Set up `GRPOConfig` with hyperparameters:
     ```python
     config = GRPOConfig(
         output_dir="./grpo_output",
         logging_dir="./grpo_logs",
         batch_size=16,              # trajectories collected before update
         mini_batch_size=4,          # for training policy/value networks
         gradient_accumulation_steps=2,
         ppo_epochs=2,               # gradient update passes per batch
         learning_rate=1e-5,
         gamma=0.4,                  # discount factor
         max_steps=100               # total GRPO update steps
     )
     ```
   - Instantiate `GRPOTrainer` with the models, config, and tokenizer

## Training Loop

The main training loop runs for `args.max_steps_train` iterations (e.g., 100). Each iteration is one GRPO update step:

### 1. Data Collection Phase

For each GRPO update step, collect `config.batch_size` (e.g., 16) trajectories:

- **For each trajectory:**
  - Select a random CUDA kernel problem
  - Set initial state:
    - `code_A_src` = Original Reference Kernel
    - `code_B_src` = Original Reference Kernel
    - `last_suggestion_A_to_B` = "" (empty)
  - Calculate baseline performance of the original kernel
  
  - **Refinement Loop** (runs for `max_steps_per_episode`, e.g., 4 times):
    - Qwen generates an optimization suggestion based on current state
    - Blackbox LLM (GPT-4o) implements the suggestion to produce `new_kernel_C_src`
    - New kernel is evaluated:
      - Compilation success (True/False)
      - Correctness (True/False)
      - Runtime performance
    - **Reward Calculation:**
      - Not compiled: large negative reward (e.g., -1.0)
      - Compiled but incorrect: medium negative reward (e.g., -0.5)
      - Correct: Base reward (0.3) + speedup_bonus
        - `speedup_bonus = baseline_time / new_kernel_time`
    - **State Update:**
      - `code_A_src` = previous `code_B_src`
      - `code_B_src` = `new_kernel_C_src`
      - `last_suggestion_A_to_B` = Qwen's suggestion
    - Store (prompt, suggestion, reward) tuple for this step

### 2. Dataset Preparation

- Flatten all trajectories into a list of (prompt, suggestion, reward) steps
- Tokenize prompts and suggestions
- Convert to `datasets.Dataset` format

### 3. GRPO Model Update

The `GRPOTrainer` updates the Qwen model:

- Iterate through dataset for `config.ppo_epochs` (e.g., 2 times)
- Process in `mini_batch_size` chunks (e.g., 4 steps)
- For each mini-batch:
  - Compute log probabilities for actions using both current and reference models
  - Calculate advantages from rewards
  - Compute policy loss using GRPO objective (balancing reward maximization with policy regularization)
  - Apply gradients after accumulation (every `gradient_accumulation_steps` mini-batches)

## Training Scale

The total number of optimization attempts is determined by:

- Refinement steps per kernel: `max_steps_per_episode` (e.g., 4)
- Trajectories per GRPO update: `grpo_collect_batch_size` (e.g., 16)
- Total GRPO updates: `max_steps_train` (e.g., 100)

Therefore:
- Total kernel optimization episodes: 100 × 16 = 1,600
- Total (prompt, suggestion, reward) samples: 100 × 16 × 4 = 6,400

## Key Components

- **Qwen Model**: The language model being trained to generate optimization suggestions
- **Reference Model**: A frozen copy of the initial Qwen model for regularization
- **Blackbox LLM**: GPT-4o that implements Qwen's suggestions into actual code
- **KernelBenchRLEnv**: Environment that handles kernel evaluation and reward calculation
- **GRPOTrainer**: Implementation of the GRPO algorithm for policy optimization

## Output

The final trained Qwen model is saved after completing `args.max_steps_train` GRPO update steps.
