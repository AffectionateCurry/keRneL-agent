import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, create_reference_model
from datasets import Dataset
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
import logging
import argparse
import os

from kernelbench_grpo_env import KernelBenchGRPOEnv # TRL Wrapper
# Assuming rl_kernel_env.py and gpt4o_coder.py are in the same src directory
# and OPENAI_API_KEY is set in the environment.

logger = logging.getLogger(__name__)

def main(args):
    # Setup output and logging directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir = Path(args.logging_dir)
    logging_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model (Qwen - suggestion model) and tokenizer
    logger.info(f"Loading Qwen model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token})")

    model_dtype = torch.float16 if args.device == "cuda" and torch.cuda.is_available() else torch.float32
    
    # For DeepSpeed, device_map might need to be None or handled by DeepSpeed config
    device_map_config = "auto" if not args.deepspeed_config else None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=model_dtype,
        device_map=device_map_config,
        trust_remote_code=True # Qwen models might require this
    )
    
    # Create reference model for GRPO
    # Ensure model is on the correct device before creating ref_model if not using device_map="auto"
    if not device_map_config and args.device == "cuda" and torch.cuda.is_available():
        model.to(args.device)
    
    ref_model = create_reference_model(model)

    # Initialize KernelBenchGRPOEnv (TRL Wrapper)
    # This wrapper will instantiate KernelBenchRLEnv internally
    rl_env_config = {
        "kernel_bench_level": args.level,
        # gpt_coder_fn is already defaulted in KernelBenchRLEnv to gpt4o_code_generator
        "blackbox_llm_model_name": args.gpt_model, # For the blackbox coder
        "max_steps_per_episode": args.max_steps_per_episode, # 4 refinement steps
        "gpu_arch_list": ["Ada"] if args.device == "cuda" and torch.cuda.is_available() else [], # Modify as needed
        "device_id": 0 if args.device == "cuda" else -1, # Assuming device 0 for cuda
        "build_cache_dir_base": str(output_dir / ".rl_cache")
    }
    # Generation kwargs for Qwen model (suggestion generation)
    generation_kwargs_qwen = {
        "max_new_tokens": args.qwen_max_new_tokens,
        "temperature": args.qwen_temperature,
        "top_p": args.qwen_top_p,
        "do_sample": args.qwen_temperature > 0.0, # Sample if temperature > 0
        "pad_token_id": tokenizer.pad_token_id,
    }

    grpo_env = KernelBenchGRPOEnv(
        rl_env_config=rl_env_config,
        generation_kwargs=generation_kwargs_qwen
        # qwen_prompt_template can be customized here if needed
    )

    # GRPO configuration
    # Total trajectories per update = num_tasks_per_batch * trajectories_per_task
    # Example: 8 tasks * 16 trajectories/task = 128 trajectories for one GRPO update cycle
    # Let's use args.num_trajectories_for_buffer as this total amount.
    # Then GRPOTrainer's batch_size is how many of these are processed by policy/value nets at once.
    
    # Effective batch size for gradient update:
    # trainer_batch_size * gradient_accumulation_steps
    # We want this to be roughly args.num_trajectories_for_buffer
    # So, gradient_accumulation_steps = args.num_trajectories_for_buffer / trainer_batch_size
    # But trl.GRPOConfig uses per_device_train_batch_size for the trainer's internal batching.
    # The 'batch_size' in GRPOConfig for collection is `config.batch_size`
    
    # Let's simplify: `GRPOConfig.batch_size` is the number of trajectories collected before an update.
    # `GRPOConfig.mini_batch_size` is the one for policy/value network forward pass.
    
    config = GRPOConfig(
        model_name=args.model_name,
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        batch_size=args.grpo_collect_batch_size, # Num trajectories to collect before an update (e.g., 128)
        mini_batch_size=args.grpo_train_mini_batch_size, # Batch size for training policy/value (e.g., 32)
        gradient_accumulation_steps=args.gradient_accumulation_steps, # Accumulate over these many mini-batches
        ppo_epochs=args.ppo_epochs, # 2 gradient steps per "GRPO batch"
        learning_rate=args.learning_rate,
        gamma=args.gamma, # Discount factor 0.4
        adap_kl_ctrl=True, # GRPO specific
        # num_train_epochs=args.num_epochs, # GRPOTrainer doesn't use this directly, relies on max_steps or outer loop
        max_steps=args.max_steps_train, # Max GRPO training steps
        save_steps=args.save_steps,
        logging_steps=1, # Log frequently
        remove_unused_columns=False,
        # deepspeed=args.deepspeed_config if args.deepspeed_config else None, # Add DeepSpeed config path
        fp16=torch.cuda.is_available() and args.device == "cuda" and model_dtype == torch.float16,
        reward_baseline = 0.0, # GRPO specific
        # num_workers = args.dataloader_num_workers # For dataloader if used
    )
    if args.deepspeed_config:
        config.deepspeed = args.deepspeed_config


    # Flatten trajectories and prepare dataset for GRPOTrainer
    # GRPOTrainer expects a list of dictionaries, where each dict is a trajectory.
    # Each trajectory dict should have "queries" (list of str), "responses" (list of str), "rewards" (list of float)
    
    # Custom dataset preparation logic
    def create_grpo_dataset(num_trajectories_to_collect: int) -> Dataset:
        collected_trajectories_data = []
        logger.info(f"Collecting {num_trajectories_to_collect} trajectories for GRPO training...")
        for _ in range(num_trajectories_to_collect):
            try:
                # generate_trajectory returns a dict like:
                # {"queries": [obs_str1, ...], "responses": [action_str1, ...], "rewards": [r1, ...]}
                traj = grpo_env.generate_trajectory(model, tokenizer)
                collected_trajectories_data.append(traj)
            except Exception as e:
                logger.error(f"Error collecting trajectory: {e}", exc_info=True)
                # Optionally, skip this trajectory or handle error
        
        # Save collected trajectories for inspection/reuse
        trajectories_save_path = output_dir / "collected_trajectories.json"
        with open(trajectories_save_path, 'w') as f:
            json.dump(collected_trajectories_data, f, indent=2)
        logger.info(f"Saved {len(collected_trajectories_data)} trajectories to {trajectories_save_path}")

        return Dataset.from_list(collected_trajectories_data)

    # Initialize GRPO trainer
    # The GRPOTrainer will use its own internal loop for collecting data using grpo_env.generate_trajectory
    # if a dataset is not directly provided or if its `train_dataset` is a generator.
    # For more control, we collect trajectories first and build a Dataset.
    
    train_dataset = create_grpo_dataset(config.batch_size) # Collect initial batch of trajectories
    
    grpo_trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        config=config,
        tokenizer=tokenizer,
        train_dataset=train_dataset 
        # GRPOTrainer.step will take this dataset and internally process it.
        # For subsequent steps, we might need to update the dataset or GRPOTrainer handles re-collection.
        # TRL usually expects the dataset to be iterated over. If more data is needed per step,
        # the outer loop should handle it.
        # For GRPO, the dataset is typically the buffer of trajectories.
    )

    # Training loop
    # GRPOTrainer's `train` method is not standard. We usually call `step` in a loop.
    # Let's assume we manage the outer loop for steps and data collection.
    logger.info("Starting GRPO training...")
    for step in range(config.max_steps): # type: ignore
        logger.info(f"GRPO Training Step {step + 1}/{config.max_steps}") # type: ignore
        
        # Collect a new batch of trajectories for this step
        # (or GRPOTrainer might do this if train_dataset is a generator or via callbacks)
        # For simplicity, let's assume `train_dataset` is used for this step's update.
        # If more data is needed, `create_grpo_dataset` should be called again.
        # However, typical PPO/GRPO implementations use a fixed buffer that's updated.
        # GRPOTrainer.step expects the current batch of queries, responses, rewards.
        
        # This part needs to align with how GRPOTrainer expects to consume data.
        # GRPOTrainer.step(queries, responses, rewards) where these are lists for the current batch.
        # The `train_dataset` provided at init is usually iterated.
        # Let's assume GRPOTrainer handles iterating `train_dataset` according to `mini_batch_size`.
        
        # If using GRPOTrainer.train() method (if it exists like in standard Trainer):
        # grpo_trainer.train() # This would handle the epochs over train_dataset
        # This is not how PPO/GRPO trainers in TRL usually work.
        
        # We need to manually iterate, collect data, and call `grpo_trainer.step`.
        # The `train_dataset` is just the initial buffer.
        # The GRPO trainer expects to get data dynamically usually.

        # Let's prepare data for one `grpo_trainer.step` call:
        # This means `config.batch_size` trajectories.
        current_batch_data = create_grpo_dataset(config.batch_size) # type: ignore
        
        queries_batch = [item["queries"] for item in current_batch_data] # This results in list of lists
        responses_batch = [item["responses"] for item in current_batch_data] # List of lists
        rewards_batch = [torch.tensor(item["rewards"], dtype=torch.float32) for item in current_batch_data] # List of Tensors

        # Flatten if generate_trajectory returns one episode at a time
        # and GRPOTrainer.step expects flat lists corresponding to (query, response, reward) tuples
        flat_queries = []
        flat_responses = []
        flat_rewards = []
        for i in range(len(queries_batch)):
            # GRPOTrainer expects queries and responses to be tokenized tensors
            # but the prompt asks for vLLM which suggests string-level interaction first.
            # Let's pass strings and let GRPOTrainer tokenize if it does.
            # The documentation shows passing lists of strings.
            
            # A trajectory has multiple steps. GRPO processes (prompt, gen, reward) for each step.
            for j in range(len(queries_batch[i])):
                flat_queries.append(queries_batch[i][j])
                flat_responses.append(responses_batch[i][j])
                # Rewards should be a flat list of scalars, not tensors yet.
                # This needs to align with how GRPOTrainer consumes it.
                # GRPOTrainer usually wants a list of Tensors for rewards.

        # This part is tricky: GRPOTrainer expects a list of rewards, usually one per (query, response) pair.
        # Let's re-evaluate: `GRPOTrainer` step takes `queries: List[torch.Tensor], responses: List[torch.Tensor], rewards: List[torch.Tensor]`
        # So, the data collection needs to produce tokenized queries and responses.
        # The `KernelBenchGRPOEnv.generate_trajectory` should probably return tokenized versions,
        # or we tokenize here. The reward list should match.

        # Simpler approach: GRPOTrainer can take a Dataset object.
        # The Dataset should yield dicts with 'input_ids', 'attention_mask' for queries,
        # and 'labels' for responses (for generation), and 'reward'.
        # This means `prepare_dataset` in the user's old `trainer.py` was on the right track.
        
        # Let's use the user's `prepare_dataset` logic from their `trainer.py` for formatting.
        # It flattens trajectories into individual (obs, action, reward) steps.
        # This `data_collator` might be implicitly handled by GRPOTrainer if dataset has right columns.
        
        # Re-using the `prepare_dataset` idea:
        all_trajectories_for_step = []
        for _ in range(config.batch_size): # type: ignore # Collect `batch_size` trajectories
             all_trajectories_for_step.append(grpo_env.generate_trajectory(model, tokenizer))
        
        processed_data_for_grpo = []
        for traj in all_trajectories_for_step:
            obs_list = traj["queries"] # Prompts to Qwen
            act_list = traj["responses"] # Suggestions from Qwen
            rew_list = traj["rewards"]
            for i in range(len(obs_list)):
                # GRPOTrainer wants tokenized input_ids for query, and labels for response
                query_tokens = tokenizer(obs_list[i], truncation=True, padding=False, max_length=args.qwen_max_prompt_len, return_tensors="pt")["input_ids"].squeeze(0)
                # Response tokens (Qwen's suggestion)
                # GRPOTrainer often expects 'labels' for the response part in SL training
                # For GRPO, it's the generated sequence.
                response_tokens = tokenizer(act_list[i], truncation=True, padding=False, max_length=args.qwen_max_new_tokens, return_tensors="pt")["input_ids"].squeeze(0)

                processed_data_for_grpo.append({
                    "input_ids": query_tokens,
                    "attention_mask": torch.ones_like(query_tokens), # Assuming simple attention mask
                    "labels": response_tokens, # The generated suggestion tokens
                    "reward": torch.tensor(rew_list[i], dtype=torch.float32)
                })
        
        if not processed_data_for_grpo:
            logger.warning("No data collected in this step. Skipping GRPOTrainer.step.")
            continue
            
        # Create a Dataset from this batch for GRPOTrainer
        step_dataset = Dataset.from_list(processed_data_for_grpo)
        
        # GRPOTrainer.train() iterates over the dataset for ppo_epochs.
        # This is more aligned with how trl.PPOTrainer works.
        stats = grpo_trainer.train(step_dataset) # Pass the dataset for this update
        logger.info(f"Step {step+1} stats: {stats}")

        if (step + 1) % config.save_steps == 0: # type: ignore
            save_path = output_dir / f"checkpoint_step_{step+1}"
            grpo_trainer.save_model(str(save_path))
            logger.info(f"Model saved to {save_path}")

    # Save final model
    final_model_path = output_dir / "final_grpo_model"
    grpo_trainer.save_model(str(final_model_path))
    logger.info(f"Training complete. Final model saved to {final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Qwen model for kernel optimization using GRPO with KernelBench.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct", help="Qwen model for suggestions.")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o", help="Blackbox LLM for code generation.")
    parser.add_argument("--level", type=int, default=1, help="KernelBench difficulty level.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for training.")
    
    parser.add_argument("--output_dir", type=str, default="./grpo_kernel_output")
    parser.add_argument("--logging_dir", type=str, default="./grpo_kernel_logs")
    
    # RL Env Params
    parser.add_argument("--max_steps_per_episode", type=int, default=4, help="Max refinement steps per trajectory.")

    # Qwen Generation Params
    parser.add_argument("--qwen_max_prompt_len", type=int, default=1536) # Max length of prompt to Qwen
    parser.add_argument("--qwen_max_new_tokens", type=int, default=256) # Max length of Qwen's suggestion
    parser.add_argument("--qwen_temperature", type=float, default=0.7)
    parser.add_argument("--qwen_top_p", type=float, default=0.9)

    # GRPO Config Params
    parser.add_argument("--grpo_collect_batch_size", type=int, default=16, help="Trajectories to collect before GRPO update (e.g., 8 tasks * 2 trajectories/task).") # Reduced for testing
    parser.add_argument("--grpo_train_mini_batch_size", type=int, default=4, help="Mini-batch size for GRPO policy/value network training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2) # (collect_batch_size / mini_batch_size) / ppo_epochs ideally
    parser.add_argument("--ppo_epochs", type=int, default=2, help="Number of PPO epochs (gradient steps) per GRPO batch.")
    parser.add_argument("--learning_rate", type=float, default=1e-5) # Adjusted from 5e-6
    parser.add_argument("--gamma", type=float, default=0.4, help="Discount factor for rewards.")
    
    # Training Loop Params
    # parser.add_argument("--num_epochs", type=int, default=3, help="Total training epochs (outer loop).") # Using max_steps instead
    parser.add_argument("--max_steps_train", type=int, default=100, help="Total GRPO training steps.") # Reduced for testing
    parser.add_argument("--save_steps", type=int, default=20)
    
    parser.add_argument("--deepspeed_config", type=str, default=None, help="Path to DeepSpeed config file.")
    # parser.add_argument("--dataloader_num_workers", type=int, default=0)


    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parsed_args = parser.parse_args()
    
    # Ensure OPENAI_API_KEY is set if gpt_model is an OpenAI model
    if "gpt" in parsed_args.gpt_model and not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set, but GPT model is selected for blackbox coder.")
        exit(1)

    main(parsed_args)