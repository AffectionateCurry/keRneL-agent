#!/usr/bin/env python3
import os
import sys
import json
import time
import random
import gc  # for garbage collection
import argparse
import yaml

import modal
from datasets import load_dataset

# ─── Paths & defaults ────────────────────────────────────────────────────────
SCRIPT_DIR                = os.path.dirname(os.path.abspath(__file__))
REPO_TOP_DIR              = os.path.dirname(SCRIPT_DIR)
LOCAL_SRC_PATH            = os.path.join(REPO_TOP_DIR, "src")
LOCAL_KERNELBENCH_DATA    = os.path.join(REPO_TOP_DIR, "KernelBench")
DEFAULT_DATASET_NAME      = "ScalingIntelligence/KernelBench"

# Qwen & ChatGPT defaults
DEFAULT_QWEN_MODEL_ID     = "Qwen/Qwen3-14B"
DEFAULT_QWEN_GPU_TYPE     = "H100"
DEFAULT_QWEN_MAX_TOKENS   = 250
DEFAULT_QWEN_TEMPERATURE  = 0.2

DEFAULT_CHAT_SERVER       = "openai"
DEFAULT_CHAT_MODEL        = "gpt-4-turbo-preview"
DEFAULT_CHAT_TEMPERATURE  = 0.0
DEFAULT_CHAT_MAX_TOKENS   = 4096

# Modal timeouts & chunking
DEFAULT_MODAL_APP_NAME    = "kernelbench_qwen_chatgpt_gen_app"
DEFAULT_QWEN_TIMEOUT      = 720
DEFAULT_ORCH_TIMEOUT      = 1200
DEFAULT_CHUNK_SIZE        = 10
DEFAULT_CHUNK_DELAY       = 30

# ─── Modal setup ─────────────────────────────────────────────────────────────
image_qwen_gen = (
    modal.Image
      .from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
      .pip_install(
          "torch>=2.1.0",
          "transformers[sentencepiece]>=4.36.0",
          "accelerate>=0.25.0",
          "bitsandbytes>=0.41.3",
          "scipy",
          "datasets",
          "openai",
          "python-dotenv",
          "pyyaml",
          "together",
          "google-generativeai",
          "anthropic",
          "utils"
      )
      # mount your code & data dirs
      .add_local_dir(local_path=LOCAL_SRC_PATH,             remote_path="/root/src")
      .add_local_dir(local_path=LOCAL_KERNELBENCH_DATA,     remote_path="/root/KernelBench")
)
app_qwen_gen    = modal.App(name=DEFAULT_MODAL_APP_NAME)
openai_secret   = modal.Secret.from_name("openai-api-key-secret")

# ─── Qwen suggestion function ─────────────────────────────────────────────────
@app_qwen_gen.function(
    image=image_qwen_gen,
    gpu=DEFAULT_QWEN_GPU_TYPE,
    timeout=DEFAULT_QWEN_TIMEOUT
)
def get_qwen_suggestions_on_modal(
    ref_kernel_code: str,
    qwen_model_id: str,
    qwen_max_new_tokens: int,
    qwen_temperature: float,
    verbose: bool
) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"[Qwen] Loading {qwen_model_id} on {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(qwen_model_id, trust_remote_code=True)
        model     = AutoModelForCausalLM.from_pretrained(
                        qwen_model_id,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        trust_remote_code=True
                    )
        model.eval()

        prompt = (
            "Analyze the following CUDA kernel code and provide 2–3 actionable optimization suggestions."
            f"\n```python\n{ref_kernel_code}\n```"
        )
        inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
                      inputs.input_ids,
                      max_new_tokens=qwen_max_new_tokens,
                      temperature=qwen_temperature or None,
                      do_sample=bool(qwen_temperature)
                  )
        suggestions = tokenizer.decode(
                          outputs[0][inputs.input_ids.shape[-1]:],
                          skip_special_tokens=True
                      )
        return suggestions.strip() or "No suggestions provided."

    except Exception as e:
        err = f"Error in Qwen Suggestor: {type(e).__name__}: {e}"
        print(err, file=sys.stderr)
        return err

    finally:
        try:
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

# ─── Orchestrator function ───────────────────────────────────────────────────
@app_qwen_gen.function(
    image=image_qwen_gen,
    secrets=[openai_secret],
    timeout=DEFAULT_ORCH_TIMEOUT
)
def orchestrate_generation_on_modal(
    problem_id: int,
    sample_id: int,
    level: int,
    dataset_src: str,
    dataset_name: str,
    qwen_model_id: str,
    qwen_gpu_type: str,
    qwen_max_new_tokens: int,
    qwen_temperature: float,
    chatgpt_server_type: str,
    chatgpt_model_name: str,
    chatgpt_temperature: float,
    chatgpt_max_tokens: int,
    verbose: bool,
    log_prompt: bool
) -> dict:
    from src.dataset import construct_kernelbench_dataset
    from src.utils   import extract_first_code, create_inference_server_from_presets

    # 1) Load reference code
    if dataset_src == "huggingface":
        ds      = load_dataset(dataset_name, trust_remote_code=True)[f"level_{level}"]
        example = next(e for e in ds if e["problem_id"] == problem_id)
        ref_code = example["code"]
    else:
        paths   = construct_kernelbench_dataset(level)
        ref_code = open(paths[problem_id - 1]).read()

    # 2) Qwen suggestions
    qwen_sugg = get_qwen_suggestions_on_modal.remote(
        ref_code,
        qwen_model_id,
        qwen_max_new_tokens,
        qwen_temperature,
        verbose
    )

    # 3) ChatGPT prompt & run
    chat_prompt = (
        "You are an expert in CUDA C++ and PyTorch. Optimize the code below."
        f"\nOriginal Code:\n{ref_code}\nSuggestions:\n{qwen_sugg}"
    )
    server = create_inference_server_from_presets(
        server_type=chatgpt_server_type,
        model_name=chatgpt_model_name,
        temperature=chatgpt_temperature,
        max_tokens=chatgpt_max_tokens,
        verbose=verbose
    )
    response  = server(chat_prompt)
    optimized = extract_first_code(response, ["python"])

    if not optimized:
        return {
            "status":     "error",
            "message":    "No valid Python code generated.",
            "problem_id": problem_id,
            "sample_id":  sample_id
        }

    return {
        "status":         "success",
        "kernel_code":    optimized,
        "qwen_suggestions": qwen_sugg,
        "problem_id":      problem_id,
        "sample_id":       sample_id
    }

# ─── Main driver ───────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Run Qwen→GPT kernel optimization")
    p.add_argument("--dataset_src",  required=True, choices=["huggingface", "local"])
    p.add_argument("--level",        type=int,    required=True)
    p.add_argument("--run_name",     required=True)
    p.add_argument("--subset",       default=None,
                   help="Optional: start,end (inclusive) of problem IDs")
    p.add_argument("--qwen_model_id",    default=DEFAULT_QWEN_MODEL_ID)
    p.add_argument("--qwen_gpu_type",    default=DEFAULT_QWEN_GPU_TYPE)
    p.add_argument("--qwen_max_new_tokens", type=int, default=DEFAULT_QWEN_MAX_TOKENS)
    p.add_argument("--qwen_temperature",    type=float, default=DEFAULT_QWEN_TEMPERATURE)
    p.add_argument("--chatgpt_model_name",  default=DEFAULT_CHAT_MODEL)
    p.add_argument("--chatgpt_temperature", type=float, default=DEFAULT_CHAT_TEMPERATURE)
    p.add_argument("--chatgpt_max_tokens",  type=int,   default=DEFAULT_CHAT_MAX_TOKENS)
    p.add_argument("--runs_dir",            default=os.path.join(REPO_TOP_DIR, "runs"))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log_prompt", action="store_true")
    p.add_argument("--modal_chunk_size", type=int, default=DEFAULT_CHUNK_SIZE,
                   help=f"Number of tasks to process in parallel per Modal starmap call (default: {DEFAULT_CHUNK_SIZE})")
    p.add_argument("--modal_inter_chunk_delay", type=int, default=DEFAULT_CHUNK_DELAY,
                   help=f"Delay in seconds between processing chunks (default: {DEFAULT_CHUNK_DELAY})")
    args = p.parse_args()

    # parse subset
    if args.subset:
        start, end = map(int, args.subset.split(",", 1))
    else:
        start, end = None, None

    # prepare run dir & save config
    run_dir = os.path.join(args.runs_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    # load metadata
    if args.dataset_src == "huggingface":
        meta = load_dataset(DEFAULT_DATASET_NAME, trust_remote_code=True)[f"level_{args.level}"]
    else:
        from src.dataset import construct_kernelbench_dataset
        meta = construct_kernelbench_dataset(args.level)

    total = len(meta)
    s, e = (start or 1), (end or total)
    ids   = list(range(s, e+1))

    tasks = [(pid, 0) for pid in ids]
    results = []

    print(f"Processing {len(tasks)} tasks in chunks of {DEFAULT_CHUNK_SIZE}")

    with modal.enable_output(), app_qwen_gen.run():
        for i in range(0, len(tasks), DEFAULT_CHUNK_SIZE):
            chunk = tasks[i : i+DEFAULT_CHUNK_SIZE]
            call_args = [
                (
                  pid, sid,
                  args.level,
                  args.dataset_src,
                  DEFAULT_DATASET_NAME,
                  args.qwen_model_id,
                  args.qwen_gpu_type,
                  args.qwen_max_new_tokens,
                  args.qwen_temperature,
                  DEFAULT_CHAT_SERVER,
                  args.chatgpt_model_name,
                  args.chatgpt_temperature,
                  args.chatgpt_max_tokens,
                  args.verbose,
                  args.log_prompt
                )
                for pid, sid in chunk
            ]
            out = orchestrate_generation_on_modal.starmap(call_args, order_outputs=True)
            results.extend(out)
            if i + DEFAULT_CHUNK_SIZE < len(tasks):
                time.sleep(DEFAULT_CHUNK_DELAY)

    # write results
    for res in results:
        if res.get("status") == "success":
            fn = f"level_{args.level}_p{res['problem_id']}_s{res['sample_id']}_kernel.py"
            with open(os.path.join(run_dir, fn), "w") as f:
                f.write(res["kernel_code"])

    print("Run complete.")

if __name__ == "__main__":
    main()
