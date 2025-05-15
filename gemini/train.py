# File: gemini/train.py

import os
import sys
from pathlib import Path
import logging
import argparse
import json
from typing import Optional

# (Optional) if you ever want to run this directly under Modal, guard the chdir:
if os.environ.get("MODAL"):
    os.chdir("/app")
# Ensure repo root is on your path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, create_reference_model
from datasets import Dataset

from gemini.kernel_grpo_env import KernelBenchGRPOEnv # Corrected: Relative to gemini package

# Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def run_training(config: dict):
    # replicate the body of your previous run_training() method
    args = argparse.Namespace(**config)

    # load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    ref_model = create_reference_model(model)
    logger.info("Models loaded.")

    # prep output dirs
    out_dir = Path(args.output_base) / args.output_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.output_base) / args.logging_dir_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # make env
    rl_env_cfg = {
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
    env = KernelBenchGRPOEnv(rl_env_cfg, gen_kwargs)
    logger.info(f"Env ready: {env.get_environment_details()}")

    # set up GRPO
    grpo_cfg = GRPOConfig(
        model_name=args.model_name,
        reward_baseline=args.reward_baseline,
        log_with="tensorboard",
        tracker_project_name=(args.wandb_project_name if args.use_wandb else None),
        tracker_kwargs=({"wandb": {"name": args.output_dir_name}} if args.use_wandb else None),
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
    trainer = GRPOTrainer(model=model, ref_model=ref_model, config=grpo_cfg, tokenizer=tokenizer)

    # training loop
    for step in range(grpo_cfg.max_steps):
        trajectories = [env.generate_trajectory(model, tokenizer) for _ in range(grpo_cfg.batch_size)]
        # flatten & tokenize
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

    # final save
    final = out_dir / "final"
    trainer.save_model(str(final))
    logger.info(f"Training complete, model at {final}")
    return str(final)

if __name__ == "__main__":
    # load JSON/YAML or defaults
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to JSON/YAML config")
    args = parser.parse_args()

    # define defaults (same as in your modal entrypoint)
    default_cfg = {
        "model_name": "Qwen/Qwen2-0.5B-Instruct",
        "gpt_model": "gpt-4o",
        "level": 1,
        "output_base": "/runs",            # for Modal, you’ll override this to /runs; locally you can set it to "./out"
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

    # override if user passed a config file
    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                if cfg_path.suffix in {".yaml", ".yml"}:
                    import yaml
                    loaded = yaml.safe_load(f)
                else:
                    loaded = json.load(f)
            default_cfg.update(loaded)

    run_training(default_cfg)
