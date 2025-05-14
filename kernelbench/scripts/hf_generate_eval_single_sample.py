import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json
import modal
import time

# Make sure API keys are loaded if needed (might not be for HF models)
from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset

from src.eval import eval_kernel_against_ref
from src.utils import set_gpu_arch, read_file, extract_first_code # Need extract_first_code

app = modal.App("hf_generate_eval_single_sample")

"""
Generates kernel using a Hugging Face model *inside* Modal, then evaluates it.
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

gpu_arch_mapping = {"L40S": ["Ada"], "H100": ["Hopper"], "A100": ["Ampere"], "L4": ["Ada"], "T4": ["Turing"], "A10G": ["Ampere"]}

# --- Updated Configuration ---
class EvalConfig(Config):
    def __init__(self):
        self.dataset_src = REQUIRED
        self.dataset_name = "ScalingIntelligence/KernelBench"
        self.level = REQUIRED
        self.problem_id = REQUIRED

        # --- Hugging Face Generator Config ---
        self.hf_model_id = "codellama/CodeLlama-7b-Instruct-hf" # Example HF model ID
        # Add generation params if needed (max_length, temp, etc.)
        self.hf_max_new_tokens = 4096
        self.hf_temperature = 0.1 # Low temp for code

        # --- Evaluation Configuration (Modal) ---
        self.eval_mode = "modal" # Fixed for this script
        self.gpu = "L40S"        # GPU for Modal generation AND evaluation

        # Logging
        self.logdir = os.path.join(REPO_TOP_DIR, "results/hf_modal_eval_logs")
        self.verbose_eval = False # Verbosity for evaluation phase
        self.log_eval_result = False
        self.log_generated_kernel = False # Log the kernel generated inside modal

    def __repr__(self):
        return (f"EvalConfig(dataset='{self.dataset_src}/{self.dataset_name}', "
                f"problem='L{self.level}-P{self.problem_id}', "
                f"generator='HF:{self.hf_model_id}', "
                f"eval='{self.eval_mode}/{self.gpu}')")

# --- Modal Setup (Needs transformers, accelerate) ---
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git", "gcc-10", "g++-10", "clang")
    .pip_install( # Add HF libraries
        "transformers>=4.34.0", # Ensure a recent version
        "accelerate>=0.22.0", # Needed for device placement
        "bitsandbytes", # Optional: for 8/4bit quantization
        "torch==2.5.0", # Keep torch consistent
        "anthropic", "numpy", "openai", "packaging", "pydra_config",
        "tqdm", "datasets", "google-generativeai", "together",
        "pytest", "ninja", "utils", "python-dotenv", "pyyaml",
    )
    # Mount src for eval and utils
    .mount(modal.Mount.from_local_dir(os.path.join(REPO_TOP_DIR, "src"), remote_path="/root/src"))
)


# --- Define a Prompt Function (can be moved to src/prompts eventually) ---
def create_hf_prompt(ref_arch_src):
     # Basic prompt, adapt as needed for your specific HF model's instruction format
     # This uses a simplified version of the original prompt structure
     prompt = f"""Optimize the following PyTorch 'Model' architecture by implementing custom CUDA kernels using `torch.utils.cpp_extension.load_inline`. Create a new PyTorch module named 'ModelNew' that is functionally equivalent but uses CUDA kernels for potential speedups.

**Original PyTorch Code ('Model'):**
```python
{ref_arch_src}

"""
     # Note: Some models might need specific instruction formatting (e.g., [INST]...[/INST])
     return prompt


@app.cls(image=image, gpu=modal.gpu.L40S(), secrets=[modal.Secret.from_dotenv()]) # Specify GPU type for the class
class HFGenerateAndEval:

    def __enter__(self):
        # Load model and tokenizer when the Modal container starts for this class instance
        # This happens ONCE per container cold start on the Modal GPU.
        print(f"Loading HF model: {self.hf_model_id}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        # Add quantization config if needed (e.g., load_in_8bit=True with bitsandbytes)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_id,
            torch_dtype=torch.bfloat16, # Adjust dtype based on GPU/model
            device_map="auto", # Use accelerate to put model on GPU(s)
            trust_remote_code=True # If required by the model
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
        print(f"Model {self.hf_model_id} loaded onto device: {self.model.device}")

    def __init__(self, hf_model_id: str, hf_max_new_tokens: int, hf_temperature: float):
        # Store config needed for generation
        self.hf_model_id = hf_model_id
        self.hf_max_new_tokens = hf_max_new_tokens
        self.hf_temperature = hf_temperature
        # __enter__ will load the model/tokenizer


    @modal.method()
    def generate_and_evaluate(self, ref_arch_src: str, verbose_eval: bool, gpu_arch_list: list[str]):
        """
        Generates kernel using the loaded HF model and evaluates it.
        """
        from src.eval import eval_kernel_against_ref
        from src.utils import set_gpu_arch, extract_first_code
        import torch # Ensure torch is imported in the method scope too

        print("--- Starting HF Generation Step ---")
        start_gen_time = time.time()

        # 1. Generate Kernel using loaded HF model
        prompt = create_hf_prompt(ref_arch_src) # Create the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generation parameters
        generate_kwargs = {
            "max_new_tokens": self.hf_max_new_tokens,
            "temperature": self.hf_temperature,
            "do_sample": self.hf_temperature > 0, # Only sample if temp > 0
            "pad_token_id": self.tokenizer.eos_token_id # Common practice
        }
        if self.hf_temperature <= 0:
             generate_kwargs["top_k"] = 1 # Force greedy if temp is 0

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        # Decode only the newly generated tokens
        raw_generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        end_gen_time = time.time()
        print(f"HF Generation took {end_gen_time - start_gen_time:.2f} seconds.")
        print("--- Raw HF Output (Preview) ---")
        print(raw_generated_text[:500] + "...")
        print("-------------------------------")

        # 2. Extract Code
        # Assuming the model follows instructions and ends the code block
        custom_cuda = extract_first_code(raw_generated_text + "```", ["python"]) # Add ``` to help regex find the end

        if custom_cuda is None or custom_cuda.strip() == "":
            print("[Error] Failed to extract valid code from HF model output.")
            # Return a failure state or raise an error
            # For simplicity, returning None, main script should handle this
            return None # Indicate generation failure

        print("--- Extracted Kernel Code (Preview) ---")
        print('\n'.join(custom_cuda.split('\n')[:15]) + "\n...")
        print("---------------------------------------")

        # Optional: Log generated kernel here if needed (e.g., to a shared volume)

        # 3. Evaluate Kernel
        print("--- Starting Evaluation Step ---")
        print(f"Setting GPU Arch for evaluation: {gpu_arch_list}")
        set_gpu_arch(gpu_arch_list)

        # Call the evaluation function (ensure it's importable)
        eval_result = eval_kernel_against_ref(
            ref_arch_src, custom_cuda, verbose=verbose_eval, measure_performance=True,
            num_correct_trials=5, num_perf_trials=100,
            # Pass the current device (Modal assigns one)
            device=self.model.device # Use the same device model is on
        )
        print("--- Evaluation Complete ---")
        return eval_result, custom_cuda # Return result and the generated code


@pydra.main(base=EvalConfig)
def main(config: EvalConfig):

    print(f"Starting HF Generate + Eval with config: {config}")

    eval_gpu_arch_list = gpu_arch_mapping.get(config.gpu)
    if not eval_gpu_arch_list:
        raise ValueError(f"GPU architecture mapping not found for evaluation GPU: {config.gpu}")
    print(f"Targeting evaluation GPU Arch: {eval_gpu_arch_list} for {config.gpu}")

    # Load Dataset (same as before)
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        from src.dataset import construct_kernelbench_dataset
        curr_level_dataset = construct_kernelbench_dataset(config.level)
    else:
        raise ValueError("dataset_src must be 'huggingface' or 'local'")

    if config.log_eval_result or config.log_generated_kernel:
        os.makedirs(config.logdir, exist_ok=True)
        pydra.save_yaml(config.to_dict(), os.path.join(config.logdir, f"main_eval_config_L{config.level}_P{config.problem_id}.yaml"))

    # Fetch Problem (same as before)
    problem_name = ""
    ref_arch_src = ""
    if config.dataset_src == "huggingface":
        problem_id_int = int(config.problem_id)
        filtered_rows = curr_level_dataset.filter(lambda x: x["problem_id"] == problem_id_int)
        if len(filtered_rows) == 0:
             raise ValueError(f"Problem ID {problem_id_int} not found in Hugging Face dataset for level {config.level}")
        curr_problem_row = filtered_rows[0]
        ref_arch_src = curr_problem_row["code"]
        problem_name = curr_problem_row["name"]
    elif config.dataset_src == "local":
        problem_idx_in_dataset = config.problem_id - 1
        if not 0 <= problem_idx_in_dataset < len(curr_level_dataset):
             raise ValueError(f"Problem ID {config.problem_id} results in invalid index for local dataset list.")
        ref_arch_path = curr_level_dataset[problem_idx_in_dataset]
        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)

    problem_number_in_name = int(problem_name.split("_")[0])
    assert problem_number_in_name == config.problem_id, \
        f"Problem number in filename ({problem_number_in_name}) does not match config problem_id ({config.problem_id})"


    # --- Call Modal Function for Generation + Evaluation ---
    print(f"Submitting Generation & Evaluation task to Modal for GPU {config.gpu}...")
    with app.run():
        # Instantiate the Modal class with HF config
        generator_evaluator = HFGenerateAndEval(
            hf_model_id=config.hf_model_id,
            hf_max_new_tokens=config.hf_max_new_tokens,
            hf_temperature=config.hf_temperature
        )
        # Call the combined method
        result = generator_evaluator.generate_and_evaluate.remote(
            ref_arch_src=ref_arch_src,
            verbose_eval=config.verbose_eval,
            gpu_arch_list=eval_gpu_arch_list
        )

        if result is None:
            print("[Error] Modal function indicated generation failure.")
            kernel_exec_result = None
            generated_kernel_code = None
        else:
            kernel_exec_result, generated_kernel_code = result


        if kernel_exec_result:
            print(f"\n--- Final Evaluation Result (L{config.level} P{config.problem_id}) ---")
            print(kernel_exec_result)
            print("--------------------------------------------------")
        else:
             print(f"\n--- Evaluation Failed or Not Performed (L{config.level} P{config.problem_id}) ---")


        # --- Logging ---
        if generated_kernel_code and config.log_generated_kernel:
             kernel_log_path = os.path.join(config.logdir, f"generated_kernel_L{config.level}_P{config.problem_id}.py")
             with open(kernel_log_path, "w") as f:
                 f.write(generated_kernel_code)
             print(f"Logged generated kernel to {kernel_log_path}")

        if kernel_exec_result and config.log_eval_result:
            log_file = os.path.join(config.logdir, f"eval_result_L{config.level}_P{config.problem_id}.json")
            result_dict = {
                "main_eval_config": config.to_dict(),
                "problem_name": problem_name,
                "eval_result": kernel_exec_result.dict() # Use pydantic's dict method
            }
            with open(log_file, "w") as f:
                 json.dump(result_dict, f, indent=2)
            print(f"Logged evaluation results to {log_file}")


if __name__ == "__main__":
    main()