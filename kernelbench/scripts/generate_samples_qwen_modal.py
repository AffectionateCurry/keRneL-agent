import os
import time
import random
import torch
import json

import pydra
from pydra import REQUIRED, Config, save_yaml

import modal
from datasets import load_dataset

# ------------------------------------------------------------------------------
# Path Setup (adjust as needed)
# ------------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_TOP_DIR = os.path.dirname(SCRIPT_DIR)

# ------------------------------------------------------------------------------
# Modal Image Definition
# ------------------------------------------------------------------------------
image_gen = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10")
    .pip_install(
        # Core libraries:
        "torch>=2.1.0",            # ensure CUDA support
        "transformers>=4.36.0",    # for Qwen 14B
        "datasets>=2.0.0",         # for load_dataset
        "python-dotenv",           # if you use .env in src/
        "utils",                   # your own utils package (if published); otherwise remove
    )
    # Mount your local source code so that `from src.dataset import ...` works:
    .add_local_python_source("src", copy=True)
    # Mount your KernelBench data folder in the container
    .add_local_dir("KernelBench", remote_path="/root/KernelBench")
)

app_gen = modal.App("kernelbench_generate_modal_qwen14b")
openai_secret = modal.Secret.from_name("openai-api-key-secret")  # Not used for Qwen, but kept here if needed

# ------------------------------------------------------------------------------
# Modal Function: Generate One Sample (using Qwen 14B locally)
# ------------------------------------------------------------------------------
@app_gen.function(
    image=image_gen,
    timeout=600,           # give enough time for Qwen to load and generate
    gpu="A10G",            # or change to whatever GPU you request in Modal
    max_containers=2
)
def generate_sample_on_modal(
    problem_id: int,
    sample_id: int,
    level: int,
    dataset_src_config: str,
    dataset_name_config: str,
    qwen_model_id: str,
    qwen_max_new_tokens: int,
    qwen_temperature: float,
    verbose_config: bool,
    run_dir_modal_path: str,
    log_prompt_config: bool
):
    """
    This function runs *inside* a Modal container. It:
      1. Loads the specified (problem_id, sample_id) from Hugging Face or local dataset.
      2. Constructs a "custom CUDA prompt" via your own prompt template function.
      3. Loads Qwen 14B (from Hugging Face) onto GPU and runs .generate(...) to get code.
      4. Returns a dict containing the generated code (and prompt if requested).
    """
    # --------------------------------------------
    # 1) Imports inside Modal container
    # --------------------------------------------
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from src.dataset import construct_kernelbench_dataset
    from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
    from src.utils import extract_first_code, read_file

    # --------------------------------------------
    # 2) (Optional) Delay for rate‐limiting jitter
    # --------------------------------------------
    base_sleep = 3.0
    jitter = random.uniform(0.5, 2.0)
    sleep_time = base_sleep + jitter
    if verbose_config:
        print(f"[Modal Gen p{problem_id}_s{sample_id}] Sleeping for {sleep_time:.2f}s before loading dataset...")
    time.sleep(sleep_time)

    # --------------------------------------------
    # 3) Load problem source code
    # --------------------------------------------
    if dataset_src_config == "huggingface":
        ds = load_dataset(dataset_name_config, trust_remote_code=True)[f"level_{level}"]
        rows = ds.filter(lambda x: x["problem_id"] == problem_id)
        if len(rows) == 0:
            return {
                "problem_id": problem_id,
                "sample_id": sample_id,
                "status": "error",
                "message": f"Problem ID {problem_id} not found in level {level}"
            }
        ref_arch_src = rows["code"][0]
        problem_name = rows["name"][0]
    else:  # local
        all_paths = construct_kernelbench_dataset(level)
        idx = problem_id - 1
        if not (0 <= idx < len(all_paths)):
            return {
                "problem_id": problem_id,
                "sample_id": sample_id,
                "status": "error",
                "message": f"Local problem ID {problem_id} out of range for level {level}"
            }
        path = all_paths[idx]
        problem_name = os.path.basename(path)
        ref_arch_src = read_file(path)

    # Warn if name/ID mismatch
    try:
        name_num = int(problem_name.split("_")[0])
        if name_num != problem_id and verbose_config:
            print(f"[Modal Gen WARNING] Name‐ID mismatch: {name_num} ≠ {problem_id}")
    except:
        pass

    # --------------------------------------------
    # 4) Build the prompt for “custom CUDA code”
    # --------------------------------------------
    custom_cuda_prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)

    # Save the prompt if requested
    if log_prompt_config:
        prompt_path = os.path.join(
            run_dir_modal_path,
            f"level_{level}_problem_{problem_id}_sample_{sample_id}_prompt.txt"
        )
        os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
        with open(prompt_path, "w") as f:
            f.write(custom_cuda_prompt)

    # --------------------------------------------
    # 5) Load Qwen 14B (once per Modal function call)
    # --------------------------------------------
    if verbose_config:
        print(f"[Modal Gen p{problem_id}_s{sample_id}] Loading Qwen model: {qwen_model_id}")

    tokenizer = AutoTokenizer.from_pretrained(qwen_model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        qwen_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()

    if verbose_config:
        print(f"[Modal Gen] Qwen loaded on device: {model.device}")

    # --------------------------------------------
    # 6) Tokenize and generate
    # --------------------------------------------
    inputs = tokenizer(
        custom_cuda_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(model.device)

    if verbose_config:
        print(f"[Modal Gen] Generating (max_new_tokens={qwen_max_new_tokens}, temperature={qwen_temperature})...")

    gen_ids = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=qwen_max_new_tokens,
        temperature=qwen_temperature if qwen_temperature > 0 else None,
        do_sample=(qwen_temperature > 0),
        pad_token_id=tokenizer.eos_token_id
    )

    # Extract only the newly generated portion:
    generated_ids = gen_ids[0][ inputs.input_ids.shape[-1] : ]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # --------------------------------------------
    # 7) Extract “python” or “cpp” code block
    # --------------------------------------------
    custom_cuda_code = extract_first_code(generated_text, ["python", "cpp"])
    if custom_cuda_code is None:
        # Write raw LLM output for debugging
        out_path = os.path.join(
            run_dir_modal_path,
            f"level_{level}_problem_{problem_id}_sample_{sample_id}_raw_output.txt"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(generated_text or "<empty>")
        return {
            "problem_id": problem_id,
            "sample_id": sample_id,
            "status": "error",
            "message": "Qwen didn’t generate a valid code block"
        }

    # --------------------------------------------
    # 8) Cleanup & Return
    # --------------------------------------------
    del model, tokenizer
    torch.cuda.empty_cache()
    gc = __import__('gc')
    gc.collect()

    if verbose_config:
        print(f"[Modal Gen] Successfully generated kernel for p{problem_id}_s{sample_id}")

    return {
        "problem_id": problem_id,
        "sample_id": sample_id,
        "status": "success",
        "kernel_code": custom_cuda_code,
        "level": level
    }


# ------------------------------------------------------------------------------
# Pydra Config Class (for local “main” invocation)
# ------------------------------------------------------------------------------
class GenerationConfigModal(Config):
    def __init__(self):
        self.dataset_src = REQUIRED
        self.dataset_name = "ScalingIntelligence/KernelBench"
        self.level = REQUIRED
        self.subset = (None, None)

        # New parameter: how many samples (k) per problem
        self.num_samples: int = 1

        # Qwen 14B parameters:
        self.qwen_model_id = "Qwen/Qwen2-14B-Instruct"
        self.qwen_max_new_tokens = 512
        self.qwen_temperature = 0.2

        self.run_name = REQUIRED
        self.runs_dir = os.path.join(REPO_TOP_DIR, "runs")
        self.verbose = False
        self.log_prompt = False


def check_kernel_exists(run_dir: str, level: int, problem_id: int, sample_id: int) -> bool:
    kernel_path = os.path.join(
        run_dir,
        f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py"
    )
    return os.path.exists(kernel_path)


@pydra.main(base=GenerationConfigModal)
def main(config: GenerationConfigModal):
    print(f"Starting Qwen-14B Modal Generation with config:\n{config}")

    # 1) Determine how many problems exist
    if config.dataset_src == "huggingface":
        local_meta = load_dataset(config.dataset_name, trust_remote_code=True)[f"level_{config.level}"]
        num_problems = len(local_meta)
    else:
        from src.dataset import construct_kernelbench_dataset
        local_paths = construct_kernelbench_dataset(config.level)
        num_problems = len(local_paths)

    # 2) Build problem_id range (same as before)
    if config.subset == (None, None):
        problem_range = range(1, num_problems + 1)
    else:
        start_id = config.subset[0] or 1
        end_id = config.subset[1] or num_problems
        assert 1 <= start_id <= end_id <= num_problems
        problem_range = range(start_id, end_id + 1)

    print(f"Will generate {config.num_samples} sample(s) for each problem in: {list(problem_range)}")

    # 3) Create run directory & save config
    run_dir = os.path.join(config.runs_dir, config.run_name)
    os.makedirs(run_dir, exist_ok=True)
    save_yaml(config.to_dict(), os.path.join(run_dir, "generation_config_modal.yaml"))

    # 4) Build list of (problem_id, sample_id) tasks
    tasks = []
    for pid in problem_range:
        for sid in range(config.num_samples):
            if not check_kernel_exists(run_dir, config.level, pid, sid):
                tasks.append((pid, sid))
            else:
                if config.verbose:
                    print(f"Skipping p{pid}_s{sid}: kernel already exists.")

    if not tasks:
        print("No new kernels to generate. Exiting.")
        return

    print(f"Found {len(tasks)} tasks. Launching Modal…")

    # 5) Invoke Modal starmap
    with modal.enable_output():
        with app_gen.run():
            args_list = [
                (
                    pid,
                    sid,
                    config.level,
                    config.dataset_src,
                    config.dataset_name,
                    config.qwen_model_id,
                    config.qwen_max_new_tokens,
                    config.qwen_temperature,
                    config.verbose,
                    run_dir,
                    config.log_prompt
                )
                for (pid, sid) in tasks
            ]

            results = list(
                generate_sample_on_modal.starmap(
                    args_list,
                    order_outputs=True
                )
            )

    # 6) Save results locally
    num_ok = 0
    num_err = 0
    for res in results:
        if res.get("status") == "success":
            pid = res["problem_id"]
            sid = res["sample_id"]
            lvl = res["level"]
            code = res["kernel_code"]

            out_path = os.path.join(
                run_dir,
                f"level_{lvl}_problem_{pid}_sample_{sid}_kernel.py"
            )
            with open(out_path, "w") as f:
                f.write(code)
            num_ok += 1
            if config.verbose:
                print(f"Saved kernel → {out_path}")
        else:
            pid = res.get("problem_id", "Unknown")
            sid = res.get("sample_id", "Unknown")
            msg = res.get("message", "<no message>")
            print(f"[ERROR] p{pid}_s{sid} → {msg}")
            num_err += 1

    print(f"\n--- Summary ---")
    print(f"Total tasks   : {len(tasks)}")
    print(f"Succeeded     : {num_ok}")
    print(f"Failed        : {num_err}")
    if num_err:
        print("Check logs above for details.")

if __name__ == "__main__":
    main()
