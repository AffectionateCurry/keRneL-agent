import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json
import modal

# added imports for ncu
from pathlib import Path
import subprocess
import uuid
import csv
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset

#from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from src.utils import extract_first_code, query_server, set_gpu_arch, read_file, create_inference_server_from_presets

app = modal.App("eval_single_sample_ncu")

"""
Generate and evaluate a single sample
Easiest way to get started, to test a single problem for experimentation or debugging
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

gpu_arch_mapping = {"L40S": ["Ada"], "H100": ["Hopper"], "A100": ["Ampere"], "L4": ["Ada"], "T4": ["Turing"], "A10G": ["Ampere"]}

# add modal volume for ncu Reports 
ncu_reports_volume = modal.Volume.from_name("kernelbench-ncu-reports", create_if_missing=True)
NCU_REPORTS_DIR = Path("/ncu_reports")

class EvalConfig(Config):
    def __init__(self):

        self.dataset_src = REQUIRED # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"


        # Problem Specification
        self.level = REQUIRED
        # NOTE: this is the logical index (problem id the problem_name)\
        self.problem_id = REQUIRED

        # Evaluation
        # local (requires a GPU), modal (cloud GPU) coming soon
        self.eval_mode = "modal"
        # Construct this from mapping from architecture name to torch cuda arch list in the future
        # you can either specify SM version or just use the name
        self.gpu = "L40S"
        self.gpu_arch = ['Ada']


        # Inference config
        self.server_type = "deepseek"
        self.model_name = "deepseek-coder"
        self.max_tokens = 4096
        self.temperature = 0.0
        
        # Logging
        self.logdir = os.path.join(REPO_TOP_DIR, "results/eval_logs")
        self.verbose = False

        self.log = False
        self.log_prompt = False
        self.log_generated_kernel = False
        self.log_eval_result = False

        # added ncu configurations 
        self.run_ncu = False # enable / disable ncu profiling
        self.ncu_metrics = [ #default
            "gpu__time_duration.sum",
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
            "lts__t_sectors_srcunit_tex_op_read.sum",
            "lts__t_sectors_srcunit_tex_op_write.sum",
            "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
            "smsp__inst_executed.sum",
        ]

    def verbose_logging(self):
        self.log = True
        self.log_prompt = True
        self.log_generated_kernel = True
        self.log_eval_result = True

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"

cuda_version = "12.4.0" # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git",
                "gcc-10",
                "g++-10",
                "clang", # note i skip a step 
                "sudo"
                )
    .pip_install(  # required to build flash-attn
        "anthropic",
        "numpy",
        "openai",
        "packaging",
        "pydra_config",
        "torch==2.5.0",
        "tqdm",
        "datasets",
        "transformers",
        "google-generativeai",
        "together",
        "pytest",
        "ninja",
        "utils",
        "python-dotenv",
        "pandas",
    )
    # Crucially, add the source code so the container can import src.*
    .copy_local_dir(local_path=os.path.join(REPO_TOP_DIR, "src"), remote_path="/root/src")
    .copy_local_file(local_path=os.path.join(REPO_TOP_DIR, "setup.py"), remote_path="/root/setup.py")
    # Install the src package inside the container
    .run_commands("cd /root && pip install -e .")
)


# helpers for parsing ncu csv 
"""Parses the raw CSV output from ncu into a list of dicts."""
def parse_ncu_csv(csv_text, metric_list):
    reader = csv.reader(csv_text.splitlines())
    data = []
    headers = []
    metric_indices = {}

    for i, row in enumerate(reader):
        if i == 0: # Header Row 1 (Metric Names)
            headers = row
            # Find indices of requested metrics - case insensitive matching
            metric_list_lower = [m.lower() for m in metric_list]
            for idx, header in enumerate(headers):
                if header.lower() in metric_list_lower:
                    metric_indices[header] = idx
            continue
        if i == 1: # Header Row 2 (Units) - skip for now
            continue
        if i == 2: # Header Row 3 (----) - skip
            continue
        if not row or len(row) < 5: # Skip empty or short rows
            continue

        # Actual data rows
        kernel_data = {"KernelName": row[4]} # Assuming Kernel Name is always column 4
        try:
            kernel_data["gpu__time_duration.sum"] = float(row[headers.index("gpu__time_duration.sum")])
        except (ValueError, IndexError):
            kernel_data["gpu__time_duration.sum"] = 0.0 # Default if missing

        for metric_name, col_index in metric_indices.items():
             if col_index < len(row):
                try:
                    # Attempt to convert to float, handle potential non-numeric values like '%'
                    value_str = row[col_index].replace('%', '').strip()
                    if value_str:
                         kernel_data[metric_name] = float(value_str)
                    else:
                         kernel_data[metric_name] = 0.0 # Handle empty strings
                except ValueError:
                    # Keep as string or default if conversion fails
                    print(f"Warning: Could not convert value '{row[col_index]}' for metric '{metric_name}' to float.")
                    kernel_data[metric_name] = 0.0 # Or keep row[col_index] if string is ok
             else:
                 kernel_data[metric_name] = 0.0 # Metric column doesn't exist in this row

        data.append(kernel_data)
    return data


@app.cls(image=image, volumes={NCU_REPORTS_DIR: ncu_reports_volume})
class EvalFunc:

    @modal.method()
    def eval_single_sample_modal(self, ref_arch_src, custom_cuda, verbose, gpu_arch):
        # --- This is the original evaluation method ---
        from src.eval import eval_kernel_against_ref
        from src.utils import set_gpu_arch
        import torch
        import os

        print("--- Running Standard Evaluation ---")
        session_id = str(uuid.uuid4())
        build_dir = f"/tmp/eval_build_{session_id}"
        os.makedirs(build_dir, exist_ok=True)

        set_gpu_arch(gpu_arch)
        result = eval_kernel_against_ref(
            ref_arch_src, custom_cuda, verbose=verbose,
            measure_performance=True, num_correct_trials=5, num_perf_trials=100,
            build_dir=build_dir # Use temp build dir
        )
        print("--- Standard Evaluation Complete ---")
        return result.model_dump() # Return dict for JSON compatibility


    @modal.method() # Specify GPU here or get from config
    def profile_kernel_with_ncu(self, ref_arch_src: str, custom_cuda_src: str, gpu_arch: list[str], metrics: list[str]):
        # --- This is the new NCU profiling method ---
        print("--- Starting NCU Profiling ---")
        from src.utils import set_gpu_arch
        import torch
        import os
        import time

        # 1. Prepare directories and unique names
        session_id = str(uuid.uuid4())
        report_filename_base = f"report_{session_id}"
        report_filename_ncu = f"{report_filename_base}.ncu-rep"
        report_path_in_volume = NCU_REPORTS_DIR / report_filename_ncu
        temp_script_path = Path(f"/tmp/profile_target_{session_id}.py")
        build_dir = f"/tmp/ncu_build_{session_id}" # Needs to be writable *inside* container

        print(f"Report will be saved to: {report_path_in_volume}")
        print(f"Temporary script: {temp_script_path}")
        print(f"Build directory: {build_dir}")

        os.makedirs(build_dir, exist_ok=True)

        # 2. Set GPU Architecture for compilation
        set_gpu_arch(gpu_arch)
        torch_arch_list = ";".join(gpu_arch)

        # 3. Create the temporary Python script content
        #    NOTE: Ensure imports inside this string refer to paths accessible in the container
        #    (e.g., /root/src/* if copied correctly)
        script_content = f"""
import torch
import os
import sys
from pathlib import Path
# Assuming src was copied to /root/src and installed via setup.py
from src.eval import load_original_model_and_inputs, load_custom_model

# Set env vars for the script's process
os.environ['TORCH_EXTENSIONS_DIR'] = '{build_dir}'
os.environ['TORCH_CUDA_ARCH_LIST'] = '{torch_arch_list}'

ref_arch_src = '''
{ref_arch_src}
'''

custom_model_src = '''
{custom_cuda_src}
'''

ref_context = {{}}
custom_context = {{}}

try:
    # === Load Ref Arch for Inputs ===
    exec(ref_arch_src, ref_context)
    Model = ref_context.get('Model')
    get_init_inputs = ref_context.get('get_init_inputs')
    get_inputs = ref_context.get('get_inputs')
    if not all([Model, get_init_inputs, get_inputs]):
        raise ValueError("Missing Model, get_init_inputs, or get_inputs in ref arch")

    # === Load Custom Kernel ===
    # Use load_custom_model which handles the build dir env var internally now
    ModelNew = load_custom_model(custom_model_src, custom_context, '{build_dir}')
    if ModelNew is None:
        raise ValueError("ModelNew not found or failed to load from custom source")

    # === Execution Logic ===
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    init_inputs = get_init_inputs()
    init_inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs]

    print("Instantiating ModelNew...")
    model_new = ModelNew(*init_inputs).cuda(device=device)
    print("ModelNew instantiated.")
    torch.cuda.synchronize() # Ensure compilation finishes if JIT

    inputs = get_inputs()
    inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]

    print("Running forward pass for NCU...")
    with torch.no_grad():
        # Warmup run (optional, NCU can also skip)
        # _ = model_new(*inputs)
        # torch.cuda.synchronize()
        # print("Warmup complete.")
        # Actual profiled run
        output = model_new(*inputs)
    torch.cuda.synchronize()
    print("Forward pass complete.")

except Exception as e:
    print(f"Error during temp script execution: {{e}}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

print("Temp script finished successfully.")
"""
        # 4. Write the temporary script
        try:
            with open(temp_script_path, "w") as f:
                f.write(script_content)
            print(f"Temporary script written successfully.")
        except Exception as e:
            print(f"Error writing temporary script: {e}")
            raise

        # 5. Execute NCU Profile command using subprocess
        
        ncu_profile_cmd = [
            "sudo", "/usr/local/cuda/bin/ncu", "--set", "full", "--import-source", "yes",
            "-o", str(report_path_in_volume),
            "--target-processes", "all",
            "-f",
            "python", str(temp_script_path)
        ]


        print(f"Running NCU Profile command: {' '.join(ncu_profile_cmd)}")

        profile_stdout, profile_stderr, profile_retcode = "", "", -1
        try:
            env = os.environ.copy()
            env['TORCH_CUDA_ARCH_LIST'] = torch_arch_list # Redundant? Maybe helpful.

            result = subprocess.run(
                ncu_profile_cmd, capture_output=True, text=True, check=False, env=env
            )
            profile_stdout, profile_stderr, profile_retcode = result.stdout, result.stderr, result.returncode

            print("--- NCU Profile Subprocess Output ---")
            print("Exit Code:", profile_retcode)
            if profile_stdout: print("STDOUT:\n", profile_stdout)
            if profile_stderr: print("STDERR:\n", profile_stderr)
            print("--- End NCU Profile Subprocess Output ---")

            if not report_path_in_volume.exists():
                 print(f"Error: NCU report file was not created at {report_path_in_volume}")
                 raise RuntimeError(f"NCU profiling failed. Check logs. Report not found.")
            print(f"NCU report successfully generated at {report_path_in_volume}")

        except FileNotFoundError:
            print("Error: 'ncu' command not found.")
            raise
        except Exception as e:
            print(f"An error occurred during NCU profile execution: {e}")
            raise

        # 6. Execute NCU Metrics command using subprocess
        metrics_str = ",".join(metrics)
        ncu_metrics_cmd = [
            "ncu", "-i", str(report_path_in_volume),
            "--csv", "--page", "raw",
            "--metrics", metrics_str
        ]
        print(f"Running NCU Metrics command: {' '.join(ncu_metrics_cmd)}")

        metrics_stdout, metrics_stderr, metrics_retcode = "", "", -1
        parsed_data = None
        try:
            result = subprocess.run(
                ncu_metrics_cmd, capture_output=True, text=True, check=False # check=False allows us to see output even on error
            )
            metrics_stdout, metrics_stderr, metrics_retcode = result.stdout, result.stderr, result.returncode

            print("--- NCU Metrics Subprocess Output ---")
            print("Exit Code:", metrics_retcode)
            if metrics_retcode == 0 and metrics_stdout:
                print("CSV Output:\n", metrics_stdout)
                # Parse the CSV data
                parsed_data = parse_ncu_csv(metrics_stdout, metrics)
                print("\nParsed Metrics Data:")
                for k_data in parsed_data:
                    print(k_data)
            else:
                 if metrics_stdout: print("STDOUT:\n", metrics_stdout)
                 if metrics_stderr: print("STDERR:\n", metrics_stderr)
                 print("Failed to extract metrics.")
            print("--- End NCU Metrics Subprocess Output ---")

        except FileNotFoundError:
            print("Error: 'ncu' command not found during metrics export.")
            raise
        except Exception as e:
            print(f"An error occurred during NCU metrics execution: {e}")
            raise
        finally:
            # Clean up the temporary script
            if temp_script_path.exists():
                try:
                    os.remove(temp_script_path)
                except OSError as e:
                    print(f"Warning: Could not remove temp script {temp_script_path}: {e}")

        # Return dict containing report path and parsed metrics
        return {
            "report_relative_path": report_filename_ncu,
            "parsed_metrics": parsed_data,
            "profile_stdout": profile_stdout,
            "profile_stderr": profile_stderr,
            "profile_retcode": profile_retcode,
            "metrics_stdout": metrics_stdout,
            "metrics_stderr": metrics_stderr,
            "metrics_retcode": metrics_retcode,
        }

@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    
    """
    Keep it simple: Generate and evaluate a single sample
    """
    print(f"Starting Eval with config: {config}")

    # Configurations
    
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]

    if config.log:
        os.makedirs(config.logdir, exist_ok=True)

        
    # Problem Checks
    num_problems = len(curr_level_dataset)
    print(f"Number of problems in Level {config.level}: {num_problems}")
    print(f"Start Generation + Evaluation for Level {config.level} Problem {config.problem_id}")

    assert config.problem_id <= num_problems, f"Problem ID {config.problem_id} out of range for Level {config.level}"


    # 1. Fetch Problem
    if config.dataset_src == "huggingface":

        curr_problem_row = curr_level_dataset.filter(lambda x: x["problem_id"] == config.problem_id)
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]

    elif config.dataset_src == "local":
        problem_idx_in_dataset = config.problem_id - 1 # due to dataset list being 0-indexed locally
        ref_arch_path = curr_level_dataset[problem_idx_in_dataset]

        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)
    # import pdb; pdb.set_trace()

    # Extract problem number from problem name (e.g. "1" from "1_Square_matrix_multiplication_.py")
    problem_number = int(problem_name.split("_")[0])
    assert problem_number == config.problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({config.problem_id})"
    
    
    # 2. Generate Sample
    # Create inference function with config parameters
    # We provide some presets in utils but you can also pass in your own, see query_server for more details
    inference_server = create_inference_server_from_presets(server_type=config.server_type,
                                                        model_name=config.model_name,
                                                        temperature=config.temperature,
                                                        max_tokens=config.max_tokens,
                                                        verbose=config.verbose, 
                                                        time_generation=True)
    


    custom_cuda_prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)
    if config.log_prompt:
        with open(os.path.join(config.logdir, f"prompt_level_{config.level}_problem_{config.problem_id}.txt"), "w") as f:
            f.write(custom_cuda_prompt)

    # Query server with constructed prompt
    custom_cuda = inference_server(custom_cuda_prompt)
    custom_cuda = extract_first_code(custom_cuda, ["python", "cpp"])
    # check LLM is able to generate custom CUDA code
    assert custom_cuda is not None, "Custom CUDA code generation failed"
    
    # this should be optional
    if config.log:
        with open(os.path.join(config.logdir, f"generated_kernel_level_{config.level}_problem_{config.problem_id}.py"), "w") as f:
            f.write(custom_cuda)

    with app.run():
     
        kernel_exec_result = EvalFunc.with_options(gpu=config.gpu)().eval_single_sample_modal.remote(ref_arch_src, custom_cuda, config.verbose, gpu_arch_mapping[config.gpu])
        
        print(f"Evaluation result for level {config.level} problem {config.problem_id}:\n{kernel_exec_result}")
        
        if config.log:
            with open(os.path.join(config.logdir, f"eval_result_level_{config.level}_problem_{config.problem_id}.txt"), "a") as f:
                f.write(f"Problem Name: {problem_name}\n")
                f.write(str(kernel_exec_result))

        # --- NCU Profiling (Conditional) ---
        if config.run_ncu:
            print("\n--- Initiating NCU Profiling on Modal ---")
            try:
                ncu_result = EvalFunc.with_options(gpu=config.gpu)() \
                            .profile_kernel_with_ncu.remote(
                                ref_arch_src,
                                custom_cuda,
                                gpu_arch_mapping[config.gpu],
                                config.ncu_metrics,
                            )
                print(f"\nNCU profiling result dict:\n{json.dumps(ncu_result, indent=2)}")

                # --- Download NCU Report ---
                if ncu_result and ncu_result.get("report_relative_path"):
                    report_relative_path = ncu_result["report_relative_path"]
                    local_report_filename = f"ncu_report_{config.level}_{config.problem_id}_{config.gpu}.ncu-rep"
                    local_report_path = Path("./").resolve() / local_report_filename # Save in CWD

                    print(f"\nAttempting to download NCU report '{report_relative_path}' from volume '{ncu_reports_volume.name}'...")
                    download_command = [
                        "modal", "volume", "get",
                        ncu_reports_volume.name,
                        report_relative_path,
                        str(local_report_path)
                    ]
                    print(f"Running: {' '.join(download_command)}")
                    try:
                        dl_result = subprocess.run(download_command, check=True, capture_output=True, text=True)
                        print(f"Successfully downloaded report to {local_report_path}")
                        print(f"STDOUT:\n{dl_result.stdout}")
                        print(f"You can now open it with: ncu-ui {local_report_path}")
                    except subprocess.CalledProcessError as e:
                        print(f"Error downloading report using modal CLI:")
                        print("STDOUT:", e.stdout)
                        print("STDERR:", e.stderr)
                    except FileNotFoundError:
                        print("Error: 'modal' command not found. Is the Modal CLI installed and in your PATH?")
                    except Exception as e_dl:
                        print(f"Unexpected error during download: {e_dl}")
                else:
                    print("NCU profiling did not return a valid report path.")

            except Exception as e_ncu:
                print(f"\nAn error occurred during NCU profiling execution: {e_ncu}")
                import traceback
                traceback.print_exc()
        else:
             print("\nSkipping NCU Profiling (run_ncu=False).")


if __name__ == "__main__":
    main()