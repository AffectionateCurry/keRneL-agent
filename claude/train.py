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

from .kernelbench_grpo_env import KernelBenchGRPOEnv
from .coder import KernelCoder
from ..llm_finetuning.src.common import app, axolotl_image, VOLUME_CONFIG


logger = logging.getLogger(__name__)

# Modal configuration
TRAINING_GPU_CONFIG = "a100:2"
SINGLE_GPU_CONFIG = "a10g:1"


@app.function(
    image=axolotl_image,
    gpu=TRAINING_GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=24 * 3600,
    secret=modal.Secret.from_name("openai-api-key"),
)
def train_grpo(config: Dict[str, Any]):
    """Main GRPO training function for Modal."""
    
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
def main():
    parser = argparse.ArgumentParser(description="Train Qwen for kernel optimization using GRPO")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o")
    
    # Environment configuration
    parser.add_argument("--kernel_level", type=int, default=1)
    parser.add_argument("--max_steps_per_episode", type=int, default=4)
    parser.add_argument("--num_correct_trials", type=int, default=5)
    parser.add_argument("--num_perf_trials", type=int, default=100)
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mini_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--max_training_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=20)
    
    # Generation configuration
    parser.add_argument("--qwen_max_new_tokens", type=int, default=256)
    parser.add_argument("--qwen_temperature", type=float, default=0.7)
    parser.add_argument("--qwen_top_p", type=float, default=0.9)
    parser.add_argument("--max_prompt_length", type=int, default=1536)
    
    # Directory configuration
    parser.add_argument("--output_dir", type=str, default="/runs/grpo_kernel_output")
    parser.add_argument("--logging_dir", type=str, default="/runs/grpo_kernel_logs")
    
    # Optional configuration
    parser.add_argument("--deepspeed_config", type=str, default=None)
    
    args = parser.parse_args()
    
    # Convert args to config dict
    config = vars(args)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Launch training on Modal
    train_grpo.remote(config)


if __name__ == "__main__":
    main()