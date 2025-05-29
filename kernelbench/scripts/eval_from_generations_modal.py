from dataclasses import dataclass
import shutil # Keep for potential future use, not directly used in this version's Modal func
import time   # Keep for potential future use
import pydra
from pydra import REQUIRED, Config, save_yaml
import json
import torch # Needed for local torch.cuda.get_device_name in Modal func error handling
import os
import modal # Import modal

from datasets import load_dataset # For local loading of dataset metadata

# REPO_TOP_DIR for local path calculations
REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Modal Setup ---
# Define paths relative to THIS script for mounting
SCRIPT_DIR_EVAL = os.path.dirname(os.path.abspath(__file__))
KERNELBENCH_PROJECT_DIR_EVAL = os.path.dirname(SCRIPT_DIR_EVAL) # kernelbench/
LOCAL_SRC_PATH_EVAL = os.path.join(KERNELBENCH_PROJECT_DIR_EVAL, "src") # kernelbench/src/

# Base CUDA image suitable for H100 (Hopper)
TAG_EVAL = "12.4.0-devel-ubuntu22.04" # Or newer if available and compatible
image_eval = (
    modal.Image.from_registry(f"nvidia/cuda:{TAG_EVAL}", add_python="3.10")
    .apt_install("git", "gcc-10", "g++-10", "clang")
    .pip_install(
        "numpy",
        "pydra_config",
        "torch==2.5.0", # Or your preferred CUDA-compatible PyTorch install command
        "ninja",
        "python-dotenv",
        "pydantic",
        "requests",
        "datasets",
        "together",
        "openai",
        "google-generativeai",
        "anthropic",
        "transformers",
        "utils"
    )
    # Add your 'src' directory.
    # If LOCAL_SRC_PATH is the path to the directory "src",
    # this will make "import src.module" work.
    # Modal typically places this in a way that it's on the PYTHONPATH.
    # The remote_path argument here specifies where the *contents* of local_path go.
    # To make `import src` work, you want the `src` directory itself at `/root/src` or similar.
    # Option A: Simple, often works if 'src' is a direct child of project root relative to script
    # .add_local_python_source("src") # This assumes 'src' can be found relative to script path by Modal

    # Option B: More explicit using the derived path LOCAL_SRC_PATH
    # This adds the *contents* of LOCAL_SRC_PATH to /root/src in the image.
    # So if LOCAL_SRC_PATH is .../kernelbench/src, then .../kernelbench/src/dataset.py
    # becomes /root/src/dataset.py.
    .add_local_dir(local_path=LOCAL_SRC_PATH_EVAL, remote_path="/root/src")
)

# Use a new app name to ensure image changes are picked up
app_eval = modal.App("kernelbench_eval_h100_v10")

# GPU_ARCH_MAPPING - used inside the Modal function
GPU_ARCH_MAPPING_MODAL = {
    "L40S": ["Ada"], "H100": ["Hopper"], "A100": ["Ampere"],
    "L4": ["Ada"], "T4": ["Turing"], "A10G": ["Ampere"],
}


@app_eval.function(
    image=image_eval,
    gpu="H100", # <--- SPECIFICALLY REQUEST H100 GPU
    timeout=900 # Increased timeout for H100 (compilation/longer runs), e.g., 15 minutes
)
def evaluate_sample_on_modal(
    problem_id: int,
    sample_id: int,
    level_config: int, # Contextual, not directly used by eval_kernel_against_ref
    run_name_config: str, # Contextual
    dataset_src_config: str, # Contextual
    dataset_name_config: str, # Contextual
    gpu_arch_name_config: str, # Should be "H100" when targeting H100
    num_correct_trials_config: int,
    num_perf_trials_config: int,
    measure_performance_config: bool,
    verbose_config: bool,
    ref_kernel_code: str,
    gen_kernel_code: str
):
    # --- Imports needed inside the Modal function ---
    from src.eval import eval_kernel_against_ref, KernelExecResult # Core eval components
    from src.utils import set_gpu_arch # For setting compilation target
    import torch # For torch.cuda.get_device_name
    import os # For os.makedirs, os.path.exists etc. if used for build_dir
    import shutil # For cleaning up build_dir

    # Set GPU architecture for compilation
    selected_gpu_arch_list = GPU_ARCH_MAPPING_MODAL.get(gpu_arch_name_config)
    if not selected_gpu_arch_list:
        if verbose_config:
            print(f"[Modal Eval ERROR] p{problem_id}_s{sample_id}: GPU arch name '{gpu_arch_name_config}' not found in mapping. Cannot proceed with correct compilation.")
        return {
            "problem_id": problem_id, "sample_id": sample_id, "status": "error",
            "compiled": False, "correctness": False, "runtime": None,
            "metadata": {"error": f"Invalid gpu_arch_name_config: {gpu_arch_name_config}"},
            "runtime_stats": None, "message": f"Invalid gpu_arch_name_config: {gpu_arch_name_config}"
        }
    
    if verbose_config:
        print(f"[Modal Eval] p{problem_id}_s{sample_id}: Setting GPU arch for compilation to {selected_gpu_arch_list} (from target '{gpu_arch_name_config}')")
    set_gpu_arch(selected_gpu_arch_list)
    
    # Temporary build directory within the Modal container
    # Ensure it's unique enough if multiple tasks could somehow run in the same container lifecycle (unlikely for .map with fresh containers)
    task_specific_build_dir = f"/tmp/kernelbench_build_p{problem_id}_s{sample_id}_{os.getpid()}"
    try:
        os.makedirs(task_specific_build_dir, exist_ok=True)
    except OSError as e:
        if verbose_config:
            print(f"[Modal Eval WARNING] p{problem_id}_s{sample_id}: Could not create temp build dir {task_specific_build_dir}: {e}. Using None for build_dir.")
        task_specific_build_dir = None

    if verbose_config:
        print(f"[Modal Eval] p{problem_id}_s{sample_id}: Evaluating with build_dir: {task_specific_build_dir or 'default'}")

    eval_result_obj = None
    try:
        eval_result_obj = eval_kernel_against_ref(
            original_model_src=ref_kernel_code,
            custom_model_src=gen_kernel_code,
            measure_performance=measure_performance_config,
            verbose=verbose_config,
            num_correct_trials=num_correct_trials_config,
            num_perf_trials=num_perf_trials_config,
            build_dir=task_specific_build_dir,
            # device will be the H100 assigned by Modal
        )
        
        if eval_result_obj is None: # eval_kernel_against_ref might return None on certain errors (e.g. lock file)
            if verbose_config:
                print(f"[Modal Eval WARNING] p{problem_id}_s{sample_id}: eval_kernel_against_ref returned None. Likely an issue like a lock file error during compilation.")
            return {
                "problem_id": problem_id, "sample_id": sample_id, "status": "error",
                "compiled": False, "correctness": False, "runtime": None,
                "metadata": {"error": "eval_kernel_against_ref returned None (e.g., compilation lock issue)"},
                "runtime_stats": None, "message": "eval_kernel_against_ref returned None"
            }

        return {
            "problem_id": problem_id, "sample_id": sample_id, "status": "success",
            "compiled": eval_result_obj.compiled,
            "correctness": eval_result_obj.correctness,
            "runtime": eval_result_obj.runtime,
            "metadata": eval_result_obj.metadata, # Should be JSON-serializable by now by src.eval
            "runtime_stats": eval_result_obj.runtime_stats
        }

    except Exception as e:
        if verbose_config:
            print(f"[Modal Eval CRITICAL ERROR] p{problem_id}_s{sample_id}: Uncaught exception during eval_kernel_against_ref: {type(e).__name__} - {e}")
        
        current_gpu_name = "N/A"
        try:
            current_gpu_name = torch.cuda.get_device_name(0) # Should be H100
        except Exception: pass

        metadata_error = {
            "error_type": type(e).__name__, "error_message": str(e), "hardware": current_gpu_name,
            "details": "Exception occurred calling eval_kernel_against_ref."
        }
        return {
            "problem_id": problem_id, "sample_id": sample_id, "status": "error",
            "compiled": False, "correctness": False, "runtime": None,
            "metadata": metadata_error, "runtime_stats": None, "message": str(e)
        }
    finally:
        if task_specific_build_dir and os.path.exists(task_specific_build_dir):
            try:
                shutil.rmtree(task_specific_build_dir)
                if verbose_config:
                    print(f"[Modal Eval] p{problem_id}_s{sample_id}: Cleaned up temp build dir: {task_specific_build_dir}")
            except Exception as e_clean:
                if verbose_config:
                    print(f"[Modal Eval WARNING] p{problem_id}_s{sample_id}: Failed to cleanup temp build dir {task_specific_build_dir}: {e_clean}")


# --- Config Class ---
class EvalConfigModal(Config):
    def __init__(self):
        self.run_name = REQUIRED
        self.dataset_src = REQUIRED
        self.dataset_name = "ScalingIntelligence/KernelBench"
        self.level = REQUIRED
        self.subset = (None, None)
        self.target_gpu_arch_name = "H100" # Default to H100, can be overridden
        self.runs_dir = os.path.join(REPO_TOP_DIR, "runs")
        self.verbose = False
        self.num_correct_trials = 5
        self.num_perf_trials = 100
        self.measure_performance = True
        self.modal_app_name = "kernelbench_eval_h100_v1" # Match app_eval name

# --- Local Helper Functions (same as before) ---
def fetch_ref_arch_from_problem_id_local(dataset_metadata_local, problem_id: int, dataset_src: str, level: int) -> str | None:
    from src.utils import read_file
    if dataset_src == "huggingface":
        curr_problem_row = dataset_metadata_local.filter(lambda x: x["problem_id"] == problem_id, num_proc=1, desc=None)
        if not curr_problem_row: return None
        return curr_problem_row["code"][0]
    elif dataset_src == "local":
        # Ensure src/dataset.py's KERNEL_BENCH_PATH is correct for local execution
        from src.dataset import get_kernelbench_level_dataset_local # Assuming you have this
        local_level_dataset = get_kernelbench_level_dataset_local(level)
        if not local_level_dataset: return None
        problem_idx_in_dataset = problem_id - 1
        if not (0 <= problem_idx_in_dataset < len(local_level_dataset)): return None
        ref_arch_path = local_level_dataset[problem_idx_in_dataset]
        return read_file(ref_arch_path)
    return None

def fetch_kernel_from_disk_local(run_dir: str, level: int, problem_id: int, sample_id: int) -> str | None:
    from src.utils import read_file
    kernel_path = os.path.join(run_dir, f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py")
    return read_file(kernel_path) if os.path.exists(kernel_path) else None

def check_if_eval_exists_local(problem_id: int, sample_id: int, eval_file_path: str) -> bool:
    if not os.path.exists(eval_file_path): return False
    try:
        with open(eval_file_path, 'r') as f:
            eval_results_data = json.load(f)
        return str(problem_id) in eval_results_data
    except (json.JSONDecodeError, IOError): return False

def add_to_eval_results_file_local(problem_id: int, sample_id: int, modal_eval_result: dict, eval_file_path: str):
    from src.eval import check_metadata_serializable_all_types # Import locally if used
    if os.path.exists(eval_file_path):
        try:
            with open(eval_file_path, 'r') as f:
                eval_results = json.load(f)
        except json.JSONDecodeError:
            eval_results = {}
            print(f"Warning: {eval_file_path} was corrupted. Starting new.")
    else:
        eval_results = {}

    metadata = modal_eval_result.get("metadata", {})
    # Ensure metadata is serializable before writing to JSON
    serializable_metadata = check_metadata_serializable_all_types(metadata)

    eval_results[str(problem_id)] = {
        'sample_id': sample_id,
        'compiled': modal_eval_result.get("compiled"),
        'correctness': modal_eval_result.get("correctness"),
        'metadata': serializable_metadata, # Use the cleaned metadata
        'runtime': modal_eval_result.get("runtime"),
        'runtime_stats': modal_eval_result.get("runtime_stats"),
        'modal_status': modal_eval_result.get("status"),
        'modal_message': modal_eval_result.get("message")
    }
    
    os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
    with open(eval_file_path, "w") as f:
        json.dump(eval_results, f, indent=2)

# --- Main Function ---
@pydra.main(base=EvalConfigModal)
def main(config: EvalConfigModal):
    print(f"Starting Modal Batch Evaluation from Generations with config: {config}")

    # Load dataset metadata locally
    if config.dataset_src == "huggingface":
        local_dataset_metadata = load_dataset(config.dataset_name)[f"level_{config.level}"]
    elif config.dataset_src == "local":
        # This needs src/dataset.py to correctly find local data when main() runs
        from src.dataset import get_kernelbench_level_dataset_local # Use your preferred getter
        local_dataset_metadata = get_kernelbench_level_dataset_local(config.level)
        if not local_dataset_metadata:
            print(f"Error: Could not load local dataset for level {config.level} for planning.")
            return
    else:
        raise ValueError(f"Unsupported dataset_src: {config.dataset_src}")

    num_problems_in_level = len(local_dataset_metadata)
    # ... (problem_id_range logic same as before) ...
    if config.subset == (None, None) or (config.subset[0] is None and config.subset[1] is None) :
        problem_id_range = range(1, num_problems_in_level + 1)
    else:
        start_id = config.subset[0] if config.subset[0] is not None else 1
        end_id = config.subset[1] if config.subset[1] is not None else num_problems_in_level
        assert 1 <= start_id <= end_id <= num_problems_in_level, \
            f"Subset range ({start_id}, {end_id}) out of range for Level {config.level} (1-{num_problems_in_level})"
        problem_id_range = range(start_id, end_id + 1)


    run_dir = os.path.join(config.runs_dir, config.run_name)
    # Save results to a file named after the target GPU architecture for clarity
    eval_file_path = os.path.join(run_dir, f"eval_results_modal_{config.target_gpu_arch_name}.json")

    if not os.path.isdir(run_dir):
        print(f"Run directory {run_dir} does not exist. Nothing to evaluate.")
        return

    if config.target_gpu_arch_name not in GPU_ARCH_MAPPING_MODAL:
        print(f"Error: target_gpu_arch_name '{config.target_gpu_arch_name}' not in GPU_ARCH_MAPPING_MODAL.")
        print(f"Available: {list(GPU_ARCH_MAPPING_MODAL.keys())}")
        return

    # Prepare arguments for Modal .map()
    map_call_args_list = []
    for problem_id_val in problem_id_range:
        sample_id_val = 0 # Fixed sample_id
        if not check_if_eval_exists_local(problem_id_val, sample_id_val, eval_file_path):
            gen_kernel_content = fetch_kernel_from_disk_local(run_dir, config.level, problem_id_val, sample_id_val)
            if gen_kernel_content:
                ref_kernel_content = fetch_ref_arch_from_problem_id_local(
                    local_dataset_metadata, problem_id_val, config.dataset_src, config.level
                )
                if ref_kernel_content:
                    map_call_args_list.append(
                        dict( # Using dict for .map, ensure Modal function unpacks correctly
                            problem_id=problem_id_val,
                            sample_id=sample_id_val,
                            level_config=config.level,
                            run_name_config=config.run_name,
                            dataset_src_config=config.dataset_src,
                            dataset_name_config=config.dataset_name,
                            gpu_arch_name_config=config.target_gpu_arch_name,
                            num_correct_trials_config=config.num_correct_trials,
                            num_perf_trials_config=config.num_perf_trials,
                            measure_performance_config=config.measure_performance,
                            verbose_config=config.verbose,
                            ref_kernel_code=ref_kernel_content,
                            gen_kernel_code=gen_kernel_content
                        )
                    )
                else:
                    if config.verbose: print(f"Skipping p{problem_id_val}_s{sample_id_val}: Could not fetch reference kernel.")
            else:
                if config.verbose: print(f"Skipping p{problem_id_val}_s{sample_id_val}: Generated kernel not found.")
        else:
            if config.verbose: print(f"Skipping p{problem_id_val}_s{sample_id_val}: Evaluation result already exists in {eval_file_path}.")
            
    if not map_call_args_list:
        print("No new kernels to evaluate or prerequisites missing. Exiting.")
        return

    print(f"Found {len(map_call_args_list)} kernels to evaluate via Modal on H100 (compiled for {config.target_gpu_arch_name}).")

    arg_list = [
        (
            d['problem_id'],
            d['sample_id'],
            d['level_config'],
            d['run_name_config'],
            d['dataset_src_config'],
            d['dataset_name_config'],
            d['gpu_arch_name_config'],
            d['num_correct_trials_config'],
            d['num_perf_trials_config'],
            d['measure_performance_config'],
            d['verbose_config'],
            d['ref_kernel_code'],
            d['gen_kernel_code']
        )
        for d in map_call_args_list
    ]

    with modal.enable_output(): # For Modal progress display
        with app_eval.run():
            # Using .map with a list of dictionaries
            eval_results_from_modal = list(evaluate_sample_on_modal.starmap(arg_list, order_outputs=True))

    # Process results (same logic as before)
    num_evaluated_successfully = 0
    num_eval_failed_modal = 0
    for result in eval_results_from_modal:
        if result and result.get("status") == "success":
            add_to_eval_results_file_local(
                result["problem_id"], result["sample_id"], result, eval_file_path
            )
            num_evaluated_successfully += 1
        elif result: # Error status from Modal function
            add_to_eval_results_file_local( # Log error results too
                result["problem_id"], result["sample_id"], result, eval_file_path
            )
            num_eval_failed_modal +=1
            print(f"Modal evaluation reported error for p{result.get('problem_id','?')} s{result.get('sample_id','?')}: {result.get('message','No message')}")
        else:
            # This case should ideally not happen if Modal .map correctly returns results or propagates exceptions.
            # If it does, it means a task might have failed silently or Modal had an internal issue.
            print("Received an unexpected None result from a Modal evaluation task.")
            # You might want to log which input caused this if possible, though .map obscures individual input-output for None.
            num_eval_failed_modal +=1

    print(f"\n--- Evaluation Summary for {config.target_gpu_arch_name} ---")
    print(f"Results saved to: {eval_file_path}")
    print(f"Total kernels targeted: {len(map_call_args_list)}")
    print(f"Tasks returning 'success' status: {num_evaluated_successfully}")
    print(f"Tasks returning 'error' status (or None): {num_eval_failed_modal}")
    if num_eval_failed_modal > 0:
         print(f"Check {eval_file_path} and Modal logs for details on evaluation errors.")

if __name__ == "__main__":
    main()