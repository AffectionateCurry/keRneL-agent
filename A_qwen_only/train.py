# train_grpo.py  – Qwen GRPO trainer (updated for new env wrappers)

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
from peft import LoraConfig

# ▶ new: import GROUP_COL, drop KernelCoder
from .kernelbench_grpo_env import KernelBenchGRPOEnv, GROUP_COL

logger = logging.getLogger(__name__)

# ───────────────────────────────── Modal config ──────────────────────────────
TRAINING_GPU_CONFIG = "H100:2"
SINGLE_GPU_CONFIG   = "a10g:1"
APP_NAME            = "example-axolotl"
ALLOW_WANDB         = os.environ.get("ALLOW_WANDB", "false").lower() == "true"

cuda_version = "12.4.0"
flavor       = "devel"
operating_sys= "ubuntu22.04"
base_cuda_tag= f"{cuda_version}-{flavor}-{operating_sys}"

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # target_modules=['q_proj', 'v_proj'], # Or other relevant modules for Qwen
)

def reward_passthrough(prompts, completions, reward, **kwargs):
    """
    This function is called by GRPOTrainer.
    'prompts' and 'completions' are generated/passed by the trainer.
    'reward' (and other columns from your step_dataset) comes from **kwargs.
    We simply return the pre-computed 'reward' values.
    """
    # 'reward' will be a list or tensor of rewards, one for each item in the batch.
    # Ensure it's returned as a list of floats.
    if isinstance(reward, torch.Tensor):
        return reward.cpu().tolist() # Ensure it's a list of Python floats
    elif isinstance(reward, list):
        return [float(r) for r in reward]
    else:
        # Handle cases where 'reward' might be a single value if batch size is 1
        # or raise an error if the format is unexpected.
        # This depends on how Dataset.from_list packages single-item lists for columns.
        # For safety, let's assume it's always a list.
        raise TypeError(f"Expected 'reward' to be a list or tensor, got {type(reward)}")

grpo_image = (
    modal.Image.from_registry(f"nvidia/cuda:{base_cuda_tag}", add_python="3.10")
    .apt_install(
        "git", "gcc-10", "g++-10", "clang", "ninja-build",
    )
    .pip_install("packaging")
    .pip_install(
        "anthropic", "numpy", "openai", "packaging", "pydra_config",
        "torch==2.5.0", "tqdm", "datasets", "transformers",
        "google-generativeai", "together", "pytest", "ninja", "python-dotenv",
        "trl>=0.8.0", "accelerate>=0.29.0", "deepspeed==0.14.4",
        "huggingface_hub>=0.20.0", "hf-transfer", "wandb",
        "gymnasium", "peft>=0.10.0", "hf_xet", "tvm", "tilelang"
    )
    .pip_install("flash-attn>=2.0.0")
    .env(
        dict(
            HUGGINGFACE_HUB_CACHE="/persistent_hf_cache",
            HF_HUB_ENABLE_HF_TRANSFER="1",
            TQDM_DISABLE="false",
        )
    )
    .entrypoint([])
)

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
        modal.Secret.from_name("my-openai-secret"),
        modal.Secret.from_dict({"ALLOW_WANDB": os.environ.get("ALLOW_WANDB", "false")}),
        *([modal.Secret.from_name("wandb")] if ALLOW_WANDB else []),
    ],
)

# ──────────────────────────────── main function ─────────────────────────────
@app.function(
    image=grpo_image,
    gpu=TRAINING_GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=24 * 3600,
)


def train_grpo(
    model_name: str = "Qwen/Qwen3-4B",
    # env / schedule
    kernel_level: int = 1,
    max_steps_per_episode: int = 4,
    trajectories_per_prompt: int = 2,
    max_prompts_per_batch: int = 8,          # ★ new
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
    # optimisation
    batch_size: int = 1,
    mini_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    ppo_epochs: int = 4,
    learning_rate: float = 1e-5,
    max_training_steps: int = 100,
    save_steps: int = 20,
    # generation
    qwen_max_new_tokens: int = 2048,
    qwen_temperature:  float = 1.0,
    qwen_top_p:         float = 0.9,
    max_prompt_length:  int   = 1536,
    # misc
    output_dir: str  = "/runs/grpo_kernel_output",
    logging_dir: str = "/runs/grpo_kernel_logs",
    deepspeed_config: str = None,
):
    # ═════════ directory setup ═════════
    output_dir   = Path(output_dir);   output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir  = Path(logging_dir);  logging_dir.mkdir(parents=True, exist_ok=True)

    # ═════════ model ═════════
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, model_max_length=32768 )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )

    # ═════════ env wrapper (no KernelCoder) ═════════
    rl_env_config = dict(
        kernel_bench_level      = kernel_level,
        max_steps_per_episode   = max_steps_per_episode,
        gpu_arch_list           = ["Ada"],
        device_id               = 0,
        num_correct_trials      = num_correct_trials,
        num_perf_trials         = num_perf_trials,
        modal_gpu_config        = SINGLE_GPU_CONFIG,
        cache_dir               = str(output_dir / "kernel_cache"),
    )

    grpo_env = KernelBenchGRPOEnv(
        rl_env_config         = rl_env_config,
        trajectories_per_prompt=trajectories_per_prompt,
        max_prompts_per_batch = max_prompts_per_batch,
        generation_kwargs     = dict(
            max_new_tokens   = qwen_max_new_tokens,
            temperature      = qwen_temperature,
            top_p            = qwen_top_p,
            do_sample        = qwen_temperature > 0.0,
            repetition_penalty = 1.2,
            pad_token_id     = tokenizer.pad_token_id,
        ),
    )

    # ═════════ trainer ═════════
    grpo_cfg = GRPOConfig(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=ppo_epochs,
        learning_rate=learning_rate,
        beta=0.04,
        max_steps=max_training_steps,
        save_steps=save_steps,
        logging_steps=1,
        fp16=torch.cuda.is_available(),
        deepspeed=deepspeed_config,
        remove_unused_columns=False,
        temperature=qwen_temperature,
        top_p=qwen_top_p,
        max_completion_length=qwen_max_new_tokens,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        processing_class=tokenizer,
        #group_column=GROUP_COL,  
        #         # ← relative-baseline key
        reward_funcs=[reward_passthrough]
    )

    # ═════════ training loop ═════════
    for step_idx in range(max_training_steps):
        logging.info(f"=== GRPO step {step_idx+1}/{max_training_steps} ===")

        trajectories = grpo_env.batch_generate_trajectories(
            agent_model=trainer.model,
            tokenizer=tokenizer,
        )
        if not trajectories:
            logging.warning("no trajectories collected; skipping")
            continue

        # flatten
        queries, responses, rewards, gids = [], [], [], []
        for traj in trajectories:
            queries   += traj["queries"]
            responses += traj["responses_for_grpo"]
            rewards   += traj["rewards"]
            gids      += traj[GROUP_COL]

        step_data = []
        for q, r, rew, gid in zip(queries, responses, rewards, gids):
            step_data.append(
                { "query": q, "response": r, "reward": float(rew), GROUP_COL: gid }
            )
        dataset = Dataset.from_list(step_data)

        trainer.train(dataset=dataset)

        if (step_idx + 1) % save_steps == 0:
            ckpt_dir = output_dir / f"checkpoint_{step_idx+1}"
            trainer.save_model(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))

        if modal.is_running_in_modal():
            VOLUME_CONFIG["/runs"].commit()

    # final save
    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    if modal.is_running_in_modal():
        VOLUME_CONFIG["/runs"].commit()


# ────────────────────────────── local entrypoint ────────────────────────────
@app.local_entrypoint()
def grpo_main(
    model_name: str = "Qwen/Qwen3-8B",
    kernel_level: int = 1,
    max_training_steps: int = 50,
    batch_size: int = 8,
    save_steps: int = 10,
):
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")
    train_grpo.remote(
        model_name=model_name,
        kernel_level=kernel_level,
        max_training_steps=max_training_steps,
        batch_size=batch_size,
        save_steps=save_steps,
    )


if __name__ == "__main__":
    grpo_main()
