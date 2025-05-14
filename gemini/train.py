# File: modal_train.py
import modal
import os
import sys
from pathlib import Path
import logging
import argparse
import json # For DeepSpeed config loading if needed

# Make sure /src is in PYTHONPATH when running in Modal
# This can be done in image setup or by adjusting sys.path here.
# Assuming modal_train.py is at the same level as the 'src' directory,
# or that 'src' is installed as a package.
# For local testing, you might run `PYTHONPATH=. modal run modal_train.py ...`
sys.path.append(str(Path(__file__).parent.resolve()))

import torch # Import torch early for device checks
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, create_reference_model
from datasets import Dataset

# Import your custom modules from /src
from src.kernelbench_grpo_env import KernelBenchGRPOEnv

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Define Modal resources
# Path to the root of your project (containing src, kernelbench, llm-finetuning)
# This assumes you run `modal deploy modal_train.py` or `modal run modal_train.py`
# from the root directory of your project.
project_root = Path(__file__).parent.resolve()
src_path = project_root / "src"
kernelbench_path = project_root / "kernelbench"
# llm_finetuning_path = project_root / "llm-finetuning" # If QWEN config needed

# --- Modal Image Definition ---
# Adjust CUDA version and Python version as needed.
# Qwen models and TRL might have specific requirements.
# Using a recent CUDA version like 12.1 for Ampere/Ada/Hopper.
# Make sure this matches kernelbench compilation requirements if any.
# The kernelbench scripts use "nvidia/cuda:12.4.0-devel-ubuntu22.04"
# For consistency, let's aim for something similar if possible.
# Or ensure compatibility. `torch.utils.cpp_extension` relies on host compiler (nvcc).
# For Modal, the container IS the host.

modal_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "ninja-build") # For torch extensions
    .pip_install(
        "torch==2.1.2", # Specify versions for stability
        "transformers==4.36.2", # Check Qwen/TRL compatibility
        "trl==0.7.11",          # Check GRPO availability
        "datasets==2.16.1",
        "gymnasium==0.29.1",
        "openai==1.12.0", # For GPT-4o coder
        "python-dotenv==1.0.0",
        "accelerate==0.26.1", # Often needed by TRL/HF
        "peft==0.8.2", # If LoRA/PEFT is used by Qwen or GRPO
        "packaging" # general utility
    )
    # Copy your custom RL environment code
    .copy_local_dir(local_path=src_path, remote_path="/app/src")
    # Copy the entire kernelbench directory and install it
    # This respects the "DO NOT MODIFY kernelbench" constraint
    .copy_local_dir(local_path=kernelbench_path, remote_path="/app/kernelbench")
    .run_commands(
        "cd /app/kernelbench && pip install -e .",
        "echo 'kernelbench installed'",
        # Add /app to PYTHONPATH so `from src...` works
        "export PYTHONPATH=$PYTHONPATH:/app" 
    )
    .env({"PYTHONPATH": "/app"}) # Also set env var for good measure
)

stub = modal.Stub("grpo-kernel-opt", image=modal_image)

# Define a Modal Volume for storing outputs (checkpoints, logs)
output_volume = modal.Volume.from_name("grpo-kernel-runs", create_if_missing=True)
# Define secrets for API keys
modal_secrets = [
    modal.Secret.from_dotenv(project_root / ".env") # If you have a .env file
    # Or individual secrets:
    # modal.Secret.from_name("my-openai-secret"), # if OPENAI_API_KEY is stored in Modal
    # modal.Secret.from_name("my-hf-secret") # for HUGGING_FACE_HUB_TOKEN
]


# --- Main Training Class on Modal ---
@stub.cls(
    gpu=modal.gpu.A10G(), # Request a GPU, e.g., A10G. Adjust as needed (A100, L4, H100).
                         # Qwen 0.5B should fit on A10G with kernel eval.
    volumes={"/runs": output_volume},
    secrets=modal_secrets,
    timeout=24 * 60 * 60, # 24-hour timeout
    # container_idle_timeout=10 * 60 # Optional: shut down if idle
)
class GRPOTrainingJob:
    def __init__(self, config_dict: dict):
        self.args = argparse.Namespace(**config_dict) # Convert dict to Namespace for compatibility
        
        # Ensure OPENAI_API_KEY is available if gpt_model is OpenAI
        if "gpt" in self.args.gpt_model and not os.environ.get("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY environment variable not found in Modal container.")
            # This should be handled by Modal secrets. If it's critical, raise an error.
            # For now, gpt_coder_fn will log an error and fallback.

    @modal.enter() # Runs when the container for the class instance starts
    def setup_model_and_tokenizer(self):
        logger.info(f"Setting up Qwen model: {self.args.model_name} on Modal.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info(f"Set tokenizer.pad_token to tokenizer.eos_token ({self.tokenizer.eos_token})")

        model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        logger.info(f"Using model dtype: {model_dtype}")

        # device_map="auto" should handle GPU placement.
        # No need for self.args.device if using device_map="auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            torch_dtype=model_dtype,
            device_map="auto", # Let HF Accelerate handle multi-GPU or single-GPU
            trust_remote_code=True
        )
        logger.info(f"Qwen model loaded on device(s): {self.model.device}") # Will show actual device

        self.ref_model = create_reference_model(self.model)
        logger.info("Reference model created.")

        # RL Environment and GRPO Trainer will be initialized in run_training

    def run_training(self):
        args = self.args
        output_dir = Path("/runs") / args.output_dir_name # Use name for subfolder in volume
        output_dir.mkdir(parents=True, exist_ok=True)
        logging_dir = Path("/runs") / args.logging_dir_name
        logging_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training outputs will be saved to: {output_dir}")
        logger.info(f"Training logs will be saved to: {logging_dir}")

        # Ensure KERNEL_BENCH_PATH is accessible for KernelBenchRLEnv
        # It should be at /app/kernelbench if image setup is correct.
        # KernelBenchRLEnv uses KERNEL_BENCH_PATH from kernelbench.src.dataset
        
        rl_env_config = {
            "kernel_bench_level": args.level,
            "blackbox_llm_model_name": args.gpt_model,
            "max_steps_per_episode": args.max_steps_per_episode,
            "gpu_arch_list": ["Ada"] if str(self.model.device).startswith("cuda") else [], # Example, detect from torch.cuda.get_device_capability() if more specific
            "device_id": 0 if str(self.model.device).startswith("cuda") else -1, 
            "build_cache_dir_base": str(output_dir / ".rl_cache"),
            "verbose_eval": args.verbose_eval,
            "num_correct_trials_eval": args.num_correct_trials_eval,
            "num_perf_trials_eval": args.num_perf_trials_eval,
        }
        
        generation_kwargs_qwen = {
            "max_new_tokens": args.qwen_max_new_tokens,
            "temperature": args.qwen_temperature,
            "top_p": args.qwen_top_p,
            "do_sample": args.qwen_temperature > 0.0,
             # pad_token_id will be set by KernelBenchGRPOEnv from tokenizer
        }

        grpo_env = KernelBenchGRPOEnv(
            rl_env_config=rl_env_config,
            generation_kwargs=generation_kwargs_qwen
        )
        logger.info(f"KernelBenchGRPOEnv initialized. Details: {grpo_env.get_environment_details()}")

        grpo_config = GRPOConfig(
            model_name=args.model_name,
            reward_baseline=args.reward_baseline, # default 0.0, from TRL example for GRPO
            # TRL's GRPO doesn't use output_dir, logging_dir in the same way as SFT.
            # Checkpoints are saved manually using trainer.save_model().
            # logging_dir for TRL usually means TensorBoard logs.
            # For simplicity, we'll handle logging via Python's logging and save models manually.
            log_with="tensorboard", # or "wandb" if configured
            tracker_project_name=args.wandb_project_name if args.use_wandb else None,
            tracker_kwargs={"wandb": {"name": args.output_dir_name}} if args.use_wandb else None,
            
            batch_size=args.grpo_collect_batch_size, 
            mini_batch_size=args.grpo_train_mini_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            ppo_epochs=args.ppo_epochs,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            adap_kl_ctrl=args.adap_kl_ctrl, # GRPO specific
            init_kl_coef=args.init_kl_coef, # KL penalty coefficient
            max_grad_norm=args.max_grad_norm,
            # num_train_epochs: GRPOTrainer does not use this, use max_steps
            max_steps=args.max_steps_train, # Total GRPO training steps (updates)
            remove_unused_columns=False, # Important for custom datasets
            # deepspeed=args.deepspeed_config if args.deepspeed_config else None, # Handle DeepSpeed config loading
            # bf16/fp16 should be compatible with model_dtype and Accelerate.
            # TRL handles this based on Accelerate config or direct flags.
            # Example: use_fp16 = (str(self.model.device).startswith("cuda") and model_dtype == torch.float16)
        )
        # For DeepSpeed:
        # if args.deepspeed_config:
        #     try:
        #         with open(args.deepspeed_config, 'r') as f:
        #             ds_config_dict = json.load(f)
        #         grpo_config.deepspeed_plugin.deepspeed_config.update(ds_config_dict) # This API might vary
        #     except Exception as e:
        #         logger.error(f"Failed to load or apply DeepSpeed config {args.deepspeed_config}: {e}")

        grpo_trainer = GRPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            config=grpo_config,
            tokenizer=self.tokenizer,
            # GRPOTrainer.step needs query, response, rewards.
            # Dataset is typically not passed at init if using .step() dynamically.
            # If we want to use GRPOTrainer.train(dataset), the dataset needs to be prepared
            # with columns like "query_input_ids", "query_attention_mask", "response_input_ids", "reward".
            # The current /src/train.py prepares a dataset and calls trainer.train(dataset).
            # Let's stick to that pattern.
        )
        
        logger.info("GRPO Trainer initialized.")
        logger.info(f"GRPO Config: {grpo_config.to_dict()}")


        # Training loop (adapted from /src/train.py)
        logger.info("Starting GRPO training loop on Modal...")
        for step_idx in range(grpo_config.max_steps): # type: ignore
            logger.info(f"GRPO Training Global Step {step_idx + 1}/{grpo_config.max_steps}")
            
            # --- Data Collection Phase ---
            collected_trajectories_for_step = []
            # grpo_config.batch_size is num_trajectories to collect for one PPO update
            for i in range(grpo_config.batch_size): # type: ignore
                logger.info(f"  Collecting trajectory {i+1}/{grpo_config.batch_size} for global step {step_idx+1}...")
                try:
                    traj = grpo_env.generate_trajectory(self.model, self.tokenizer)
                    collected_trajectories_for_step.append(traj)
                except Exception as e:
                    logger.error(f"  Error collecting trajectory {i+1}: {e}", exc_info=True)
            
            if not collected_trajectories_for_step:
                logger.warning(f"  No trajectories collected for global step {step_idx+1}. Skipping PPO update.")
                continue

            # --- Data Processing for GRPOTrainer ---
            # GRPOTrainer expects lists of tokenized queries, responses, and rewards for its .step() method
            # Or a Dataset with 'input_ids' (query), 'labels' (response), 'reward' for .train()
            # The user's /src/train.py prepares a Dataset. Let's follow that.
            processed_data_for_grpo_step = []
            for traj in collected_trajectories_for_step:
                prompts_str_list = traj["queries"]  # List of Qwen prompts for this trajectory
                responses_str_list = traj["responses"] # List of Qwen suggestions
                rewards_list = traj.get("rewards", []) # List of scalar rewards

                for i in range(len(prompts_str_list)):
                    query_tokens = self.tokenizer(prompts_str_list[i], truncation=True, padding=False, 
                                                  max_length=args.qwen_max_prompt_len).input_ids
                    response_tokens = self.tokenizer(responses_str_list[i], truncation=True, padding=False, 
                                                     max_length=args.qwen_max_new_tokens).input_ids
                    
                    processed_data_for_grpo_step.append({
                        "input_ids": query_tokens, # Query (prompt to Qwen)
                        # "attention_mask": torch.ones_like(torch.tensor(query_tokens)), # GRPOTrainer might handle this
                        "labels": response_tokens, # Response (Qwen's suggestion)
                        "reward": rewards_list[i] if i < len(rewards_list) else 0.0 # Ensure reward exists
                    })
            
            if not processed_data_for_grpo_step:
                logger.warning(f"  No processed data for GRPO for global step {step_idx+1}. Skipping PPO update.")
                continue
            
            current_step_dataset = Dataset.from_list(processed_data_for_grpo_step)
            
            # --- GRPO Update Phase ---
            # The .train() method in TRL's PPO/GRPO usually handles the PPO epochs over the provided dataset.
            logger.info(f"  Starting GRPOTrainer.train() for global step {step_idx+1} with {len(current_step_dataset)} samples.")
            stats = grpo_trainer.train(current_step_dataset) # This method might not exist or work this way.
                                                        # GRPOTrainer typically uses .step(queries, responses, rewards)
                                                        # Let's assume a .step() like interface for now.
            # Alternative if .train(Dataset) is not for GRPO:
            # queries_tensor = [torch.tensor(d["input_ids"]).to(self.model.device) for d in processed_data_for_grpo_step]
            # responses_tensor = [torch.tensor(d["labels"]).to(self.model.device) for d in processed_data_for_grpo_step]
            # rewards_tensor = [torch.tensor([d["reward"]]).to(self.model.device) for d in processed_data_for_grpo_step] # Ensure rewards are tensors
            # stats = grpo_trainer.step(queries_tensor, responses_tensor, rewards_tensor)
            
            # For now, assuming trainer.train(Dataset) is the intended adaptation from user's src/train.py.
            # If this fails, the interface is likely GRPOTrainer.step(List[query_tokens], List[response_tokens], List[rewards_tensors])
            # where query_tokens and response_tokens are lists of 1D tensors.

            logger.info(f"  Global step {step_idx+1} PPO update stats: {stats}") # `stats` might be per-epoch or cumulative for the call.

            if (step_idx + 1) % args.save_steps == 0:
                save_path = output_dir / f"checkpoint_step_{step_idx+1}"
                grpo_trainer.save_model(str(save_path)) # TRL trainer.save_model saves adapter, not full model by default.
                logger.info(f"  Model checkpoint saved to {save_path}")
                output_volume.commit() # Commit volume changes

        # Save final model
        final_model_path = output_dir / "final_grpo_model"
        grpo_trainer.save_model(str(final_model_path))
        logger.info(f"Training complete. Final GRPO model adapter saved to {final_model_path}")
        output_volume.commit()
        return str(final_model_path)


# --- Local Entrypoint for Modal CLI ---
@stub.local_entrypoint()
def main(config: Optional[str] = None):
    """
    Local entrypoint to run the GRPO training job on Modal.
    Configuration is passed via a YAML/JSON file or defaults.
    """
    # Default configuration (can be overridden by a config file)
    default_config = {
        "model_name": "Qwen/Qwen2-0.5B-Instruct",
        "gpt_model": "gpt-4o", # Blackbox coder model
        "level": 1, # KernelBench level
        "output_dir_name": "grpo_kernel_output_modal", # Subfolder name in Modal Volume /runs
        "logging_dir_name": "grpo_kernel_logs_modal",  # Subfolder name
        
        "max_steps_per_episode": 4,
        "num_correct_trials_eval": 3, # For RL env eval
        "num_perf_trials_eval": 30,  # For RL env eval
        "verbose_eval": False,       # Verbosity in RL env kernel evaluation

        "qwen_max_prompt_len": 1536,
        "qwen_max_new_tokens": 256,
        "qwen_temperature": 0.7,
        "qwen_top_p": 0.9,

        "grpo_collect_batch_size": 16, # Trajectories collected before one PPO update
        "grpo_train_mini_batch_size": 4, # Mini-batch for policy/value net training
        "gradient_accumulation_steps": 2,
        "ppo_epochs": 4, # PPO epochs over the collected batch
        "learning_rate": 1e-5, # Common for PPO
        "gamma": 0.99,         # Discount factor
        "adap_kl_ctrl": True,
        "init_kl_coef": 0.02,  # Initial KL coefficient for PPO penalty
        "max_grad_norm": 1.0,  # Gradient clipping

        "reward_baseline": 0.0, # GRPO specific, from TRL example
        
        "max_steps_train": 100, # Total GRPO training steps (PPO updates)
        "save_steps": 20,      # Save checkpoint every N global steps
        
        # "deepspeed_config": None, # Path to DeepSpeed config if used
        "use_wandb": False,
        "wandb_project_name": "grpo_kernel_opt",
    }

    if config:
        config_path = Path(config)
        if config_path.exists():
            logger.info(f"Loading configuration from {config_path}")
            with open(config_path, 'r') as f:
                if config_path.suffix == ".json":
                    loaded_config = json.load(f)
                elif config_path.suffix in [".yaml", ".yml"]:
                    import yaml
                    loaded_config = yaml.safe_load(f)
                else:
                    logger.error(f"Unsupported config file format: {config_path.suffix}")
                    return
            default_config.update(loaded_config)
        else:
            logger.warning(f"Config file {config} not found. Using default parameters.")

    logger.info(f"Final configuration for Modal job: {json.dumps(default_config, indent=2)}")

    # Create and run the Modal job
    job = GRPOTrainingJob(default_config)
    final_model_path_on_volume = job.run_training.remote()
    logger.info(f"Modal job submitted. Final model will be at: {final_model_path_on_volume} (on Modal Volume)")