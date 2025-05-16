import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import modal
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, create_reference_model
from datasets import Dataset

from claude.kernelbench_grpo_env import KernelBenchGRPOEnv
from claude.coder import KernelCoder
# from llm_finetuning.src.common import app, VOLUME_CONFIG, axolotl_image

logger = logging.getLogger(__name__)

# Modal configuration
TRAINING_GPU_CONFIG = "a100:2"
SINGLE_GPU_CONFIG = "a10g:1"

APP_NAME = "example-axolotl"
ALLOW_WANDB = os.environ.get("ALLOW_WANDB", "false").lower() == "true"


# Create our own image with necessary dependencies
""" grpo_image = (
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
)  """

cuda_version = "12.4.0" # For torch 2.5.0, CUDA 12.1+ is typical. 12.4 is very new.
                         # Consider CUDA 12.1 or 12.2 for broader compatibility if 12.4 causes issues.
                         # PyTorch 2.5 official binaries are built with CUDA 11.8 and 12.1.
                         # Using a newer CUDA in the image (12.4) should generally be fine with torch 2.5 built for 12.1.
flavor = "devel"
operating_sys = "ubuntu22.04"
base_cuda_tag = f"{cuda_version}-{flavor}-{operating_sys}"

# --- Define the combined image ---
grpo_image = (
    modal.Image.from_registry(f"nvidia/cuda:{base_cuda_tag}", add_python="3.10")
    .apt_install(
        "git",
        "gcc-10",  # For C++ compilation
        "g++-10",  # For C++ compilation
        "clang",   # Often useful for some C++ tools or linters
        "ninja-build" # For PyTorch C++ extensions (like apex, or custom CUDA kernels in kernelbench)
    )
    .pip_install(
        # --- KernelBench Dependencies (from your original image) ---
        "anthropic",
        "numpy",        # Pin to a specific version if you encounter issues
        "openai",
        "packaging",
        "pydra_config",
        "torch==2.5.0", # Your chosen torch version
        "tqdm",
        "datasets",     # Pin to a version compatible with transformers/trl
        "transformers", # Pin to a version compatible with torch 2.5.0 and trl
        "google-generativeai",
        "together",
        "pytest",       # Optional for runtime, good for dev/testing image builds
        "ninja",        # PyPI ninja, distinct from apt ninja-build. Both can be useful.
                        # torch.utils.cpp_extension uses the one it finds.
        # "utils",      # Make sure this is the intended PyPI package, or if it's a local module, handle it via .add_local_...
        "python-dotenv",

        # --- GRPO Fine-tuning Dependencies (inspired by axolotl & TRL needs) ---
        "trl>=0.8.0", # TRL 0.8.0+ is generally compatible with newer transformers/torch. Pin more specifically if needed.
                       # Example: "trl==0.8.6"
        "accelerate>=0.29.0", # Choose a version compatible with your torch/transformers
                             # Example: "accelerate==0.29.3"
        "deepspeed==0.14.4", # Your pinned version, seems reasonable for torch 2.5.0
        "huggingface_hub>=0.20.0", # For model hub interactions
        "hf-transfer",         # For faster HF downloads
        "wandb",               # If you use it for logging
        "gymnasium",           # For KernelBenchRLEnv
        "peft>=0.10.0",        # Often a dependency for TRL/Axolotl for LoRA etc.
                              # Example: "peft==0.10.0"
        # Ensure all versions are compatible. A good starting point:
        # "torch==2.5.0", (already there)
        # "transformers==4.41.2", # Latest as of checking, good for torch 2.5
        # "datasets==2.19.2",
        # "accelerate==0.31.0",
        # "peft==0.11.1",
        # "trl==0.9.4",
        # "deepspeed==0.14.4", (already there)
    )
    # Set environment variables (similar to axolotl_image)
    .env(
        dict(
            HUGGINGFACE_HUB_CACHE="/persistent_hf_cache", # Use a clean, volume-mounted path
            HF_HUB_ENABLE_HF_TRANSFER="1",
            TQDM_DISABLE="false", # Optional: disables tqdm progress bars in logs
            # AXOLOTL_NCCL_TIMEOUT="60", # This was for Axolotl, may not be needed if not using Axolotl directly
                                       # but can be kept if using accelerate/deepspeed with NCCL.
        )
    )
    
    # Clear default entrypoint from base CUDA image
    .entrypoint([])
)

# --- Define Volume Configuration (can be in the same file or imported) ---
# This should be THE place where volumes are defined if this image is central.
# Ensure the cache path matches HUGGINGFACE_HUB_CACHE env var.
VOLUME_CONFIG = {
    "/persistent_hf_cache": modal.Volume.from_name(
        "kernel-agent-shared-hf-cache", create_if_missing=True
    ),
    "/runs": modal.Volume.from_name(
        "kernel-agent-shared-runs", create_if_missing=True
    ),
}


app = modal.App(
    APP_NAME,
    secrets=[
        modal.Secret.from_name("my-huggingface-secret"),
        modal.Secret.from_dict({"ALLOW_WANDB": os.environ.get("ALLOW_WANDB", "false")}),
        *([modal.Secret.from_name("wandb")] if ALLOW_WANDB else []),
    ],
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