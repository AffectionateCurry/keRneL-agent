import pydra
from pydra import REQUIRED, Config, save_yaml
import os
import sys
import torch # May not be strictly needed here but good to keep consistent
import json
from dataclasses import dataclass
import modal
import time
import random

from datasets import load_dataset

# Assuming these are in your src directory and will be added to Modal image
# from src.dataset import construct_kernelbench_dataset
# from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
# from src.utils import extract_first_code, create_inference_server_from_presets, read_file, maybe_multithread
# ^-- maybe_multithread will be replaced by Modal's parallelism

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Modal Setup ---
# Define the Modal Image (can be shared with eval script if dependencies are similar)
# Using a generic CUDA version for now, adjust if specific version needed for generation tools
# or if generation itself needs GPU (unlikely for just OpenAI calls)
image_gen = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10")
    .pip_install(
        "torch", "datasets", "openai", "pydra_config", "python-dotenv", "anthropic", "google-generativeai", "together", "utils", "transformers"
        # Add other LLM client libraries if needed
    )
    .add_local_python_source("src", copy=True) # Add your 'src' directory
    .add_local_dir("KernelBench", remote_path="/root/KernelBench")
)

app_gen = modal.App("kernelbench_generate_modal_v14")
openai_secret = modal.Secret.from_name("openai-api-key-secret") # USE YOUR SECRET NAME

@app_gen.function(image=image_gen, secrets=[openai_secret], timeout=300, max_containers=5) # Timeout for LLM call
def generate_sample_on_modal(
    problem_id: int,
    sample_id: int,
    level: int,
    dataset_src_config: str,
    dataset_name_config: str,
    server_type_config: str,
    model_name_config: str,
    temperature_config: float,
    max_tokens_config: int,
    verbose_config: bool,
    run_dir_modal_path: str,
    log_prompt_config: bool
):
    # --- Code from original generate_sample_single ---
    # These imports happen inside the Modal function now
    from datasets import load_dataset
    from src.dataset import construct_kernelbench_dataset
    from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
    from src.utils import extract_first_code, create_inference_server_from_presets, read_file
    from openai import RateLimitError

    base_sleep_duration = 5.0  # e.g., 5 seconds base sleep
    jitter = random.uniform(1.0, 5.0) # Add 1-5 second of jitter
    actual_sleep_duration = base_sleep_duration + jitter

    if verbose_config:
        print(f"[Modal Gen p{problem_id}_s{sample_id}] Intentionally sleeping for {actual_sleep_duration:.2f}s before API call...")
    time.sleep(actual_sleep_duration)

    # 1. Fetch Problem
    if dataset_src_config == "huggingface":
        # Load dataset inside the Modal function if not passed directly (can be large)
        dataset_modal = load_dataset(dataset_name_config)[f"level_{level}"]
        # Filtering on the full dataset can be slow.
        # Consider passing the specific problem's ref_arch_src if feasible
        # or ensuring problem_id is unique and filtering is efficient.
        curr_problem_row = dataset_modal.filter(lambda x: x["problem_id"] == problem_id, desc=None)
        if not curr_problem_row:
            print(f"[Modal Gen ERROR] Problem ID {problem_id} not found in level {level} of {dataset_name_config}")
            return {"problem_id": problem_id, "sample_id": sample_id, "status": "error", "message": "Problem ID not found"}
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]
    elif dataset_src_config == "local":
        # This assumes the 'local' dataset structure can be replicated or accessed
        # by the Modal function. If it relies on local paths not in the image,
        # this part needs careful handling (e.g., mounting a Volume or passing content).
        # For simplicity, let's assume construct_kernelbench_dataset works if src is available.
        dataset_modal = construct_kernelbench_dataset(level)
        problem_idx_in_dataset = problem_id - 1
        if not (0 <= problem_idx_in_dataset < len(dataset_modal)):
            print(f"[Modal Gen ERROR] Problem ID {problem_id} (index {problem_idx_in_dataset}) out of range for local dataset level {level}")
            return {"problem_id": problem_id, "sample_id": sample_id, "status": "error", "message": "Problem ID out of range for local dataset"}
        ref_arch_path = dataset_modal[problem_idx_in_dataset]
        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path) # read_file needs to work in Modal context
    else:
        return {"problem_id": problem_id, "sample_id": sample_id, "status": "error", "message": f"Unknown dataset_src: {dataset_src_config}"}


    problem_number_from_name = int(problem_name.split("_")[0])
    if problem_number_from_name != problem_id:
        print(f"[Modal Gen WARNING] Mismatch: problem_name number {problem_number_from_name} vs problem_id {problem_id}")
        # Continue but log this, could indicate issues.

    # Construct Prompt
    custom_cuda_prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)

    # Create inference server (inside Modal function)
    inference_server = create_inference_server_from_presets(
        server_type=server_type_config,
        model_name=model_name_config,
        temperature=temperature_config,
        max_tokens=max_tokens_config,
        verbose=verbose_config
    )

    # Query server
    if verbose_config:
        print(f"[Modal Gen] Querying LLM for problem {problem_id}, sample {sample_id}")
    custom_cuda_str = inference_server(custom_cuda_prompt)
    custom_cuda = extract_first_code(custom_cuda_str, ["python", "cpp"])

    if custom_cuda is None:
        print(f"[Modal Gen ERROR] Custom CUDA code generation failed for problem {problem_id}")
        # Optionally save the raw LLM output for debugging
        raw_output_path = os.path.join(run_dir_modal_path, f"level_{level}_problem_{problem_id}_sample_{sample_id}_raw_llm_output.txt")
        os.makedirs(os.path.dirname(raw_output_path), exist_ok=True)
        with open(raw_output_path, "w") as f:
            f.write(custom_cuda_str or "None")
        return {"problem_id": problem_id, "sample_id": sample_id, "status": "error", "message": "LLM did not generate valid code block"}


    if verbose_config:
        print(f"[Modal Gen] Generated sample {sample_id} for problem {problem_id}: {problem_name}")

    # Store to a path that Modal can write to.
    # The main script will later ensure this is in the correct local 'runs' structure.
    # For now, we assume run_dir_modal_path is a temporary path within the Modal execution if needed,
    # or directly the final path if using a shared volume.
    # For simplicity, let's assume this path is directly the final one if a NetworkFileSystem is used.
    # Otherwise, the generated code string needs to be returned and saved by the main script.

    # To keep it simple for now: return the generated code and prompt
    # The main script will handle saving.
    
    kernel_content = custom_cuda
    prompt_content = custom_cuda_prompt if log_prompt_config else None

    return {
        "problem_id": problem_id,
        "sample_id": sample_id,
        "status": "success",
        "kernel_code": kernel_content,
        "prompt_code": prompt_content,
        "level": level
    }

# --- Original Script Structure (adapted) ---

class GenerationConfigModal(Config): # Renamed to avoid conflict if in same file for testing
    def __init__(self):
        self.dataset_src = REQUIRED
        self.dataset_name = "ScalingIntelligence/KernelBench"
        self.level = REQUIRED
        self.subset = (None, None)
        self.run_name = REQUIRED
        # self.num_workers = 1 # Modal handles parallelism via .map() or multiple .remote() calls
        # self.api_query_interval = 0.0 # Modal handles rate limiting if needed, or backoff in function
        self.server_type = "openai" # Defaulting to openai as per discussion
        self.model_name = "gpt-4.1"  # Defaulting
        self.max_tokens = 4096
        self.temperature = 0.0
        self.runs_dir = os.path.join(REPO_TOP_DIR, "runs")
        self.verbose = False
        # self.store_type = "local" # Implicitly local via Modal saving results
        self.log_prompt = False
        # Add any new Modal-specific configs, e.g., Modal app name
        self.modal_app_name = "kernelbench_generate_modal"


def check_kernel_exists(run_dir: str, level: int, problem_id: int, sample_id: int) -> bool:
    kernel_path = os.path.join(run_dir, f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py")
    return os.path.exists(kernel_path)

@pydra.main(base=GenerationConfigModal)
def main(config: GenerationConfigModal):
    print(f"Starting Modal Batch Generation with config: {config}")

    # Dataset Configurations (loaded locally for determining scope)
    if config.dataset_src == "huggingface":
        local_dataset_metadata = load_dataset(config.dataset_name)[f"level_{config.level}"]
    elif config.dataset_src == "local":
        from src.dataset import construct_kernelbench_dataset # Local import for this part
        local_dataset_metadata = construct_kernelbench_dataset(config.level)
    else:
        raise ValueError(f"Unsupported dataset_src: {config.dataset_src}")

    num_problems_in_level = len(local_dataset_metadata)

    if config.subset == (None, None) or (config.subset[0] is None and config.subset[1] is None) :
        problem_id_range = range(1, num_problems_in_level + 1)
    else:
        start_id = config.subset[0] if config.subset[0] is not None else 1
        end_id = config.subset[1] if config.subset[1] is not None else num_problems_in_level
        assert 1 <= start_id <= end_id <= num_problems_in_level, \
            f"Subset range ({start_id}, {end_id}) out of range for Level {config.level} (1-{num_problems_in_level})"
        problem_id_range = range(start_id, end_id + 1)

    print(f"Targeting 1 sample each for level {config.level} problems: {list(problem_id_range)}")

    run_dir = os.path.join(config.runs_dir, config.run_name)
    os.makedirs(run_dir, exist_ok=True)
    save_yaml(config.to_dict(), os.path.join(run_dir, "generation_config_modal.yaml"))

    tasks_for_modal = []
    for problem_id in problem_id_range:
        sample_id = 0 # Fixed as per original script
        if not check_kernel_exists(run_dir, config.level, problem_id, sample_id):
            tasks_for_modal.append(
                (problem_id, sample_id) # Tuple of (problem_id, sample_id)
            )
        else:
            if config.verbose:
                print(f"Skipping problem {problem_id} sample {sample_id}, kernel already exists.")

    if not tasks_for_modal:
        print("No new kernels to generate. Exiting.")
        return

    print(f"Found {len(tasks_for_modal)} problems to generate kernels for via Modal.")

    # Run Modal functions

    with modal.enable_output():
        with app_gen.run():
            modal_call_args = [] # This is your list of correctly formed dictionaries
            for p_id, s_id in tasks_for_modal:
                current_call_kwargs = dict(
                    problem_id=p_id,
                    sample_id=s_id,
                    level=config.level,
                    dataset_src_config=config.dataset_src,
                    dataset_name_config=config.dataset_name,
                    server_type_config=config.server_type,
                    model_name_config=config.model_name,
                    temperature_config=config.temperature,
                    max_tokens_config=config.max_tokens,
                    verbose_config=config.verbose,
                    run_dir_modal_path=run_dir,
                    log_prompt_config=config.log_prompt
                )
                modal_call_args.append(current_call_kwargs)

            if not modal_call_args:
                print("modal_call_args is empty. No tasks to map.")
            else:
                print(f"First item in modal_call_args: {modal_call_args[0]}")

            # Use .map with the list of dictionaries
            # 1) build a list of tuples in the same order as your function signature
            args_list = [
                (
                    d['problem_id'],
                    d['sample_id'],
                    d['level'],
                    d['dataset_src_config'],
                    d['dataset_name_config'],
                    d['server_type_config'],
                    d['model_name_config'],
                    d['temperature_config'],
                    d['max_tokens_config'],
                    d['verbose_config'],
                    d['run_dir_modal_path'],
                    d['log_prompt_config'],
                )
                for d in modal_call_args
            ]

            # 2) use starmap to unpack each tuple into positional args
            results = list(
                generate_sample_on_modal.starmap(
                args_list,
                order_outputs=True,
                )
        )


    num_generated_successfully = 0
    num_failed = 0
    for result in results:
        if result and result.get("status") == "success":
            p_id = result["problem_id"]
            s_id = result["sample_id"]
            lvl = result["level"]
            kernel_code = result["kernel_code"]
            prompt_code = result.get("prompt_code")

            # Save the kernel locally
            kernel_path = os.path.join(run_dir, f"level_{lvl}_problem_{p_id}_sample_{s_id}_kernel.py")
            with open(kernel_path, "w") as f:
                f.write(kernel_code)
            num_generated_successfully += 1
            if config.verbose:
                print(f"Successfully generated and saved kernel for problem {p_id} sample {s_id}")

            if prompt_code:
                prompt_path = os.path.join(run_dir, f"level_{lvl}_problem_{p_id}_sample_{s_id}_prompt.txt")
                with open(prompt_path, "w") as f:
                    f.write(prompt_code)
        else:
            p_id = result.get("problem_id", "Unknown")
            s_id = result.get("sample_id", "Unknown")
            message = result.get("message", "No message")
            print(f"Failed to generate kernel for problem {p_id} sample {s_id}: {message}")
            num_failed += 1

    print(f"\n--- Generation Summary ---")
    print(f"Total problems targeted by Modal: {len(tasks_for_modal)}")
    print(f"Successfully generated and saved: {num_generated_successfully}")
    print(f"Failed generations: {num_failed}")
    if num_failed > 0:
        print("Please check logs for details on failures.")

if __name__ == "__main__":
    main()