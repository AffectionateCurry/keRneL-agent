import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import modal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, create_reference_model
from datasets import Dataset

from claude.kernelbench_grpo_env import KernelBenchGRPOEnv
from claude.coder import KernelCoder
from llm_finetuning.src.common import app, VOLUME_CONFIG, axolotl_image

logger = logging.getLogger(__name__)

# Modal configuration
TRAINING_GPU_CONFIG = "a100:2"
SINGLE_GPU_CONFIG = "a10g:1"

# Create our own image with necessary dependencies
grpo_image = (
    axolotl_image
    .pip_install(
        "transformers",
        "trl>=0.17.0",
        "datasets",
        "torch==2.5.0",
        "openai",
        "gymnasium",
        "together",
        "google-generativeai",
        "pytest",
        "ninja",
        "utils",
        "python-dotenv",
        "tqdm",
        "anthropic",
        "numpy",
        "packaging",
        "pydra_config",
        "deepspeed==0.14.4"
    )
) 


@app.function(
    image=grpo_image,
    gpu=TRAINING_GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=24 * 3600,
)
def train_grpo(
    model_name: str = "Qwen/Qwen2-0.5B-Instruct",
    gpt_model: str = "gpt-4o",
    kernel_level: int = 1,
    max_steps_per_episode: int = 4,
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
    batch_size: int = 16,
    mini_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    ppo_epochs: int = 2,
    learning_rate: float = 1e-5,
    gamma: float = 0.4,
    max_training_steps: int = 100,
    save_steps: int = 20,
    qwen_max_new_tokens: int = 256,
    qwen_temperature: float = 0.7,
    qwen_top_p: float = 0.9,
    max_prompt_length: int = 1536,
    output_dir: str = "/runs/grpo_kernel_output",
    logging_dir: str = "/runs/grpo_kernel_logs",
    deepspeed_config: str = None,
):
    """Main GRPO training function for Modal."""
    
    # Create config from parameters
    config = {
        "model_name": model_name,
        "gpt_model": gpt_model,
        "kernel_level": kernel_level,
        "max_steps_per_episode": max_steps_per_episode,
        "num_correct_trials": num_correct_trials,
        "num_perf_trials": num_perf_trials,
        "batch_size": batch_size,
        "mini_batch_size": mini_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "ppo_epochs": ppo_epochs,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "max_training_steps": max_training_steps,
        "save_steps": save_steps,
        "qwen_max_new_tokens": qwen_max_new_tokens,
        "qwen_temperature": qwen_temperature,
        "qwen_top_p": qwen_top_p,
        "max_prompt_length": max_prompt_length,
        "output_dir": output_dir,
        "logging_dir": logging_dir,
        "deepspeed_config": deepspeed_config,
    }
    
    # Setup directories  
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging_dir = Path(config["logging_dir"])
    logging_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Qwen model and tokenizer
    logger.info(f"Loading model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    ref_model = create_reference_model(model)
    
    # Initialize KernelCoder with GPT-4o
    kernel_coder = KernelCoder(model_name=config["gpt_model"])
    
    # Initialize GRPO environment
    rl_env_config = {
        "kernel_bench_level": config["kernel_level"],
        "kernel_coder": kernel_coder,
        "max_steps_per_episode": config["max_steps_per_episode"],
        "gpu_arch_list": ["Ada"],
        "device_id": 0,
        "num_correct_trials": config["num_correct_trials"],
        "num_perf_trials": config["num_perf_trials"],
        "modal_gpu_config": SINGLE_GPU_CONFIG,
        "cache_dir": str(output_dir / "kernel_cache"),
    }
    
    grpo_env = KernelBenchGRPOEnv(
        rl_env_config=rl_env_config,
        generation_kwargs={
            "max_new_tokens": config["qwen_max_new_tokens"],
            "temperature": config["qwen_temperature"],
            "top_p": config["qwen_top_p"],
            "do_sample": config["qwen_temperature"] > 0.0,
        }
    )
    
    # GRPO configuration
    grpo_config = GRPOConfig(
        model_name=config["model_name"],
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        batch_size=config["batch_size"],
        mini_batch_size=config["mini_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        ppo_epochs=config["ppo_epochs"],
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        max_steps=config["max_training_steps"],
        save_steps=config["save_steps"],
        logging_steps=1,
        fp16=torch.cuda.is_available(),
        deepspeed=config.get("deepspeed_config"),
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        config=grpo_config,
        tokenizer=tokenizer,
    )
    
    # Training loop
    logger.info("Starting GRPO training...")
    
    for step in range(config["max_training_steps"]):
        logger.info(f"Training step {step + 1}/{config['max_training_steps']}")
        
        # Collect trajectories
        trajectories = grpo_env.batch_generate_trajectories(
            agent_model=model,
            tokenizer=tokenizer,
            num_trajectories=config["batch_size"]
        )
        
        # Save trajectories for debugging
        if step % config["save_steps"] == 0:
            traj_path = output_dir / f"trajectories_step_{step}.json"
            with open(traj_path, "w") as f:
                json.dump(trajectories, f, indent=2)
        
        # Prepare data for GRPO
        all_queries = []
        all_responses = []
        all_rewards = []
        
        for traj in trajectories:
            all_queries.extend(traj["queries"])
            all_responses.extend(traj["responses"])
            all_rewards.extend(traj["rewards"])
        
        # Create dataset for this step
        step_data = []
        for q, r, reward in zip(all_queries, all_responses, all_rewards):
            query_tokens = tokenizer(
                q,
                truncation=True,
                max_length=config["max_prompt_length"],
                return_tensors="pt"
            )["input_ids"].squeeze(0)
            
            response_tokens = tokenizer(
                r,
                truncation=True,
                max_length=config["qwen_max_new_tokens"],
                return_tensors="pt"
            )["input_ids"].squeeze(0)
            
            step_data.append({
                "input_ids": query_tokens,
                "attention_mask": torch.ones_like(query_tokens),
                "labels": response_tokens,
                "reward": torch.tensor(reward, dtype=torch.float32)
            })
        
        if not step_data:
            logger.warning("No data collected in this step. Skipping...")
            continue
        
        step_dataset = Dataset.from_list(step_data)
        
        # Train on this batch
        stats = trainer.train(step_dataset)
        logger.info(f"Step {step + 1} stats: {stats}")
        
        # Save checkpoints
        if (step + 1) % config["save_steps"] == 0:
            checkpoint_dir = output_dir / f"checkpoint_{step + 1}"
            trainer.save_model(str(checkpoint_dir))
            logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Log metrics
        VOLUME_CONFIG["/runs"].commit()
    
    # Save final model
    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    logger.info(f"Training complete. Final model saved to {final_dir}")
    
    # Final volume commit
    VOLUME_CONFIG["/runs"].commit()


@app.local_entrypoint()
def grpo_main(
    model_name: str = "Qwen/Qwen2-0.5B-Instruct",
    gpt_model: str = "gpt-4o",
    kernel_level: int = 1,
    max_training_steps: int = 100,
    batch_size: int = 8,
    save_steps: int = 20,
):  
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    config = locals() 
    train_grpo.remote(**config)


if __name__ == "__main__":
    grpo_main()