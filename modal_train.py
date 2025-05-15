# File: modal_train.py
import sys
from pathlib import Path
import os # Import os after sys.path modifications if it's used for path ops early

# --- START PYTHONPATH MODIFICATION FOR LOCAL MODAL PARSING ---
# Get the project root directory (where modal_train.py is)
_PROJECT_ROOT = Path(__file__).parent.resolve()

# Path to the directory that contains the 'kernelbench' code,
# and within 'kernelbench', the 'src' directory that pip will install.
_KERNELBENCH_ROOT_DIR = _PROJECT_ROOT / "kernelbench"

# For local parsing: to make 'from src.dataset' work from within 'gemini' module,
# Python needs to find a 'src' package. We tell it that KERNELBENCH_ROOT_DIR
# is a place where it might find a top-level 'src' directory.
# This mimics how pip install -e from kernelbench/ would make kernelbench/src/ available.
if str(_KERNELBENCH_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_KERNELBENCH_ROOT_DIR))

# Also, ensure the project root itself is in sys.path so 'import gemini' works.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# --- END PYTHONPATH MODIFICATION ---

# Now, the rest of your imports
import logging # Moved standard library imports after sys.path manipulation
import json
from typing import Optional
import modal

# Import the pure training function from your gemini package
from gemini.train import run_training # This should now work if gemini is in /app

# Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__) # Use __name__ for module-specific logger

# Build Modal image
modal_image = (
    modal.Image
      .from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
      .apt_install("git", "build-essential", "ninja-build")
      .pip_install(
          "torch>=2.3.0", # Ensure compatibility with CUDA 12.1
          "transformers", # Keep versions pinned
          "trl",
          "dataset",
          "gymnasium",
          "openai",
          "python-dotenv",
          "accelerate",
          "peft",
          "packaging",
          "pyyaml", # For reading YAML configs
      )
      .copy_local_dir(local_path=".", remote_path="/app")
      .run_commands(
          "cd /app/kernelbench && pip install -e .", # Install kernelbench from /app/kernelbench
          "echo 'kernelbench installed from /app/kernelbench'",
      )
)

stub = modal.Stub("grpo-kernel-opt", image=modal_image)
output_vol = modal.Volume.from_name("grpo-kernel-runs", create_if_missing=True)
# Ensure your .env file is at the project root (same directory as modal_train.py)
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

    @modal.method()
    def run(self):
        logger.info(f"Starting training job with config: {self.cfg}")
        self.cfg["output_base"] = "/runs"
        final_model_path = run_training(self.cfg)
        output_vol.commit()
        logger.info(f"Training finished. Final model path reported: {final_model_path}")
        return final_model_path

@stub.local_entrypoint()
def main(config: Optional[str] = None):
    current_cfg = {
        "model_name": "Qwen/Qwen2-0.5B-Instruct",
        "gpt_model": "gpt-4o",
        "level": 1,
        "output_dir_name": f"grpo_run_{Path(config).stem if config else 'default'}",
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
        "learning_rate": 2e-5,
        "gamma": 0.99,
        "adap_kl_ctrl": True,
        "init_kl_coef": 0.02,
        "max_grad_norm": 1.0,
        "reward_baseline": 0.0,
        "max_steps_train": 100,
        "save_steps": 20,
        "use_wandb": False,
        "wandb_project_name": "grpo_kernel_optimization",
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
            current_cfg.update(user_cfg)
            logger.info(f"Loaded and merged configuration from: {config}")
        else:
            logger.error(f"Specified config file '{config}' not found. Using defaults.")

    job_runner = GRPOTrainingJob(cfg=current_cfg)
    model_ref_future = job_runner.run.remote()
    logger.info(f"Submitted Modal job. Call ID: {model_ref_future.object_id}")
    logger.info("You can monitor the run in the Modal dashboard.")