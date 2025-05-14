# File: gemini/train.py
import os
import sys
from pathlib import Path
import logging
import argparse
import json
from typing import Optional

# If running inside Modal, switch into /app
if os.environ.get("MODAL"):
    os.chdir("/app")

# Ensure project root is on PYTHONPATH for imports
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, create_reference_model
from datasets import Dataset

# Import your custom env from the same package
import os, sys
from pathlib import Path
import logging, json
from typing import Optional
import modal

# In‐container, switch into /app
if os.environ.get("MODAL"):
    os.chdir("/app")

# Make repo root importable
sys.path.insert(0, str(Path(__file__).parent.resolve()))

# Bring in the pure training logic
from gemini.train import run_training
# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def run_training(config: dict) -> str:
    args = argparse.Namespace(**config)

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model = create_reference_model(model)
    logger.info("Models loaded.")

    # Prepare output dirs
    base = Path(args.output_base)
    out_dir = base / args.output_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    # RL environment configuration
    rl_cfg = {
        "kernel_bench_level": args.level,
        "blackbox_llm_model_name": args.gpt_model,
        "max_steps_per_episode": args.max_steps_per_episode,
        "gpu_arch_list": ["Ada"],
        "device_id": 0,
        "build_cache_dir_base": str(out_dir / ".rl_cache"),
        "verbose_eval": args.verbose_eval,
        "num_correct_trials_eval": args.num_correct_trials_eval,
        "num_perf_trials_eval": args.num_perf_trials_eval,
    }
    gen_kwargs = {
        "max_new_tokens": args.qwen_max_new_tokens,
        "temperature": args.qwen_temperature,
        "top_p": args.qwen_top_p,
        "do_sample": args.qwen_temperature > 0.0,
    }
    env = KernelBenchGRPOEnv(rl_cfg, gen_kwargs)
    logger.info(f"Env initialized: {env.get_environment_details()}")

    # GRPO setup
    grpo_cfg = GRPOConfig(
        model_name=args.model_name,
        reward_baseline=args.reward_baseline,
        log_with="tensorboard",
        tracker_project_name=(args.wandb_project_name if args.use_wandb else None),
        tracker_kwargs=( {"wandb": {"name": args.output_dir_name}} if args.use_wandb else None ),
        batch_size=args.grpo_collect_batch_size,
        mini_batch_size=args.grpo_train_mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ppo_epochs=args.ppo_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        adap_kl_ctrl=args.adap_kl_ctrl,
        init_kl_coef=args.init_kl_coef,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps_train,
        remove_unused_columns=False,
    )
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        config=grpo_cfg,
        tokenizer=tokenizer,
    )

    # Training loop
    for step in range(grpo_cfg.max_steps):
        trajectories = [env.generate_trajectory(model, tokenizer) for _ in range(grpo_cfg.batch_size)]
        data = []
        for traj in trajectories:
            for q, r, rew in zip(traj["queries"], traj["responses"], traj.get("rewards", [])):
                q_ids = tokenizer(q, truncation=True, padding=False, max_length=args.qwen_max_prompt_len).input_ids
                r_ids = tokenizer(r, truncation=True, padding=False, max_length=args.qwen_max_new_tokens).input_ids
                data.append({"input_ids": q_ids, "labels": r_ids, "reward": rew})
        ds = Dataset.from_list(data)
        stats = trainer.train(ds)
        logger.info(f"Step {step+1}/{grpo_cfg.max_steps} stats: {stats}")
        if (step + 1) % args.save_steps == 0:
            ckpt = out_dir / f"ckpt_{step+1}"
            trainer.save_model(str(ckpt))
    # Final checkpoint
    final = out_dir / "final"
    trainer.save_model(str(final))
    logger.info(f"Training finished; model at {final}")
    return str(final)

if __name__ == "__main__":
    # Argument parsing for local runs
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to JSON/YAML config")
    args = parser.parse_args()

    # Default config (mirrors modal defaults)
    default_cfg = {
        "model_name": "Qwen/Qwen2-0.5B-Instruct",
        "gpt_model": "gpt-4o",
        "level": 1,
        "output_base": ".",  # local out folder
        "output_dir_name": "grpo_out",
        "logging_dir_name": "grpo_logs",
        "max_steps_per_episode": 4,
        "num_correct_trials_eval": 3,
        "num_perf_trials_eval": 30,
        "verbose_eval": False,
        "qwen_max_prompt_len": 1536,
        "qwen_max_new_tokens": 256,
        "qwen_temperature": 0.7,
        "qwen_top_p": 0.9,
        "grpo_collect_batch_size": 16,
        "grpo_train_mini_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "ppo_epochs": 4,
        "learning_rate": 1e-5,
        "gamma": 0.99,
        "adap_kl_ctrl": True,
        "init_kl_coef": 0.02,
        "max_grad_norm": 1.0,
        "reward_baseline": 0.0,
        "max_steps_train": 100,
        "save_steps": 20,
        "use_wandb": False,
        "wandb_project_name": "grpo_kernel_opt",
    }
    # Override from config file
    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                if cfg_path.suffix in {".yaml", ".yml"}:
                    import yaml
                    user_cfg = yaml.safe_load(f)
                else:
                    user_cfg = json.load(f)
            default_cfg.update(user_cfg)
    run_training(default_cfg)


# File: modal_train.py
import os
import sys
from pathlib import Path
import logging
import json
from typing import Optional
import modal

# Inside Modal container, switch to /app
if os.environ.get("MODAL"):
    os.chdir("/app")

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.resolve()))

# Import the pure training function
from gemini.train import run_training

# Logger
t_logging = logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Build Modal image
modal_image = (
    modal.Image
      .from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
      .apt_install("git", "build-essential", "ninja-build")
      .pip_install(
          "torch==2.1.2",
          "transformers==4.36.2",
          "trl==0.7.11",
          "datasets==2.16.1",
          "gymnasium==0.29.1",
          "openai==1.12.0",
          "python-dotenv==1.0.0",
          "accelerate==0.26.1",
          "peft==0.8.2",
          "packaging",
          "google.generativeai",
          "anthropic",
      )
      .copy_local_dir(local_path=".", remote_path="/app")
      .run_commands(
          "cd /app/kernelbench && pip install -e .",
          "echo 'kernelbench installed'",
          "export PYTHONPATH=$PYTHONPATH:/app",
      )
      .env({"PYTHONPATH": "/app"})
)

stub = modal.Stub("grpo-kernel-opt", image=modal_image)
# Volume for outputs
output_vol = modal.Volume.from_name("grpo-kernel-runs", create_if_missing=True)
# Secrets for API keys
modal_secrets = [modal.Secret.from_dotenv(Path(__file__).parent / ".env")]

@stub.cls(
    gpu=modal.gpu.A10G(),
    volumes={"/runs": output_vol},
    secrets=modal_secrets,
    timeout=24 * 60 * 60,
)
class GRPOTrainingJob:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    @modal.enter()
    def enter(self):
        pass

    def run(self):
        # set output base to the Modal volume
        self.cfg["output_base"] = "/runs"
        return run_training(self.cfg)

@stub.local_entrypoint()
def main(config: Optional[str] = None):
    # default config
    default_cfg = {
        "model_name": "Qwen/Qwen2-0.5B-Instruct",
        "gpt_model": "gpt-4o",
        "level": 1,
        "output_base": "/runs",
        "output_dir_name": "grpo_out",
        "logging_dir_name": "grpo_logs",
        "max_steps_per_episode": 4,
        "num_correct_trials_eval": 3,
        "num_perf_trials_eval": 30,
        "verbose_eval": False,
        "qwen_max_prompt_len": 1536,
        "qwen_max_new_tokens": 256,
        "qwen_temperature": 0.7,
        "qwen_top_p": 0.9,
        "grpo_collect_batch_size": 16,
        "grpo_train_mini_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "ppo_epochs": 4,
        "learning_rate": 1e-5,
        "gamma": 0.99,
        "adap_kl_ctrl": True,
        "init_kl_coef": 0.02,
        "max_grad_norm": 1.0,
        "reward_baseline": 0.0,
        "max_steps_train": 100,
        "save_steps": 20,
        "use_wandb": False,
        "wandb_project_name": "grpo_kernel_opt",
    }
    if config:
        cfg_path = Path(config)
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                if cfg_path.suffix in {".yaml", ".yml"}:
                    import yaml
                    user_cfg = yaml.safe_load(f)
                else:
                    user_cfg = json.load(f)
            default_cfg.update(user_cfg)
    job = GRPOTrainingJob(default_cfg)
    model_ref = job.run.remote()
    logger.info(f"Submitted job; final model at: {model_ref}")
