import gymnasium as gym
import numpy as np
import torch
import os
import hashlib
import shutil
import re
from typing import List, Tuple, Dict, Any, Callable, Optional

# KernelBench specific imports
# Ensure KernelBench root is in PYTHONPATH or this script is in the root
from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref, KernelExecResult
from src.utils import set_gpu_arch, read_file
from scripts.generate_baseline_time import measure_program_time

# Define REPO_TOP_DIR for default cache paths, etc.
# This assumes the script is in the KernelBench repo root.
try:
    REPO_TOP_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError: # Fallback if __file__ is not defined (e.g. in interactive interpreter)
    REPO_TOP_DIR = os.getcwd()

# Default GPT-4o coder placeholder
def default_gpt4o_coder(problem_ref_src: str, current_kernel_src: str, suggestion: str) -> str:
    """
    A placeholder for the GPT-4o coding function.
    It should take the original problem's source, current kernel source, and a textual suggestion,
    and return a new kernel string.
    IMPORTANT: Replace this with your actual GPT-4o interaction logic.
    """
    print(f"[Default GPT-4o Coder] Warning: Using placeholder. Suggestion: '{suggestion}'")
    # This naive version appends the suggestion as a comment.
    # A slightly more advanced placeholder might assume `suggestion` is the new code if it looks like code.
    # For example:
    # if "```python" in suggestion:
    #     match = re.search(r"```python\n(.*?)\n```", suggestion, re.DOTALL)
    #     if match:
    #         return match.group(1)
    return current_kernel_src + f"\n# LLM-CODER-APPLIED-SUGGESTION: {suggestion}\n"

class KernelBenchRLEnv(gym.Env):
    """
    An RL Environment Wrapper for KernelBench.

    State: (problem_description, current_kernel_code, optimization_history)
    Action: An optimization suggestion string from the Qwen model.
    Reward: Based on correctness and performance improvement.
    """
    metadata = {'render_modes': [], 'render_fps': 1} # render not applicable

    def __init__(self,
                 kernel_bench_level: int,
                 gpt4o_coder_fn: Callable[[str, str, str], str] = default_gpt4o_coder,
                 problems_subset_indices: Optional[List[int]] = None,
                 max_steps_per_episode: int = 5,
                 gpu_arch: List[str] = ["Ada"], # Example, adjust to your hardware
                 device_id: int = 0,
                 num_correct_trials: int = 5,
                 num_perf_trials: int = 100, # For reference timing and final eval
                 kernel_build_dir_base: str = os.path.join(REPO_TOP_DIR, "kernelbench_rl_cache"),
                 correctness_reward_weight: float = 0.3,
                 penalty_non_compilation: float = -1.0,
                 penalty_incorrect: float = -0.5,
                 clear_cache_on_start: bool = True
                 ):
        super().__init__()

        self.kernel_bench_level = kernel_bench_level
        self.gpt4o_coder_fn = gpt4o_coder_fn
        self.max_steps_per_episode = max_steps_per_episode
        
        if not torch.cuda.is_available() and device_id >= 0 :
            print(f"Warning: CUDA device {device_id} requested but CUDA not available. Defaulting to CPU.")
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            if device_id >= torch.cuda.device_count():
                print(f"Warning: CUDA device_id {device_id} out of range. Found {torch.cuda.device_count()} devices. Using cuda:0.")
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device(f"cuda:{device_id}")
        else: # CPU
             self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")


        if self.device.type == 'cuda':
            set_gpu_arch(gpu_arch) # This sets an environment variable
            self.gpu_arch = gpu_arch
        else:
            self.gpu_arch = [] # No GPU architecture if on CPU

        self.num_correct_trials = num_correct_trials
        self.num_perf_trials = num_perf_trials
        self.kernel_build_dir_base = kernel_build_dir_base
        if clear_cache_on_start and os.path.exists(self.kernel_build_dir_base):
            print(f"Clearing kernel build cache at {self.kernel_build_dir_base}")
            shutil.rmtree(self.kernel_build_dir_base)
        os.makedirs(self.kernel_build_dir_base, exist_ok=True)

        self.correctness_reward_weight = correctness_reward_weight
        self.penalty_non_compilation = penalty_non_compilation
        self.penalty_incorrect = penalty_incorrect

        self.dataset = construct_kernelbench_dataset(level=self.kernel_bench_level)
        if not self.dataset:
            raise ValueError(f"Could not load KernelBench dataset for level {self.kernel_bench_level}")

        if problems_subset_indices:
            self.problem_indices_to_run = [idx for idx in problems_subset_indices if 0 <= idx < len(self.dataset)]
            if not self.problem_indices_to_run:
                 raise ValueError("Problem subset indices are out of bounds or result in an empty set.")
        else:
            self.problem_indices_to_run = list(range(len(self.dataset)))
        
        self._problem_iterator_idx = -1 # Iterates through self.problem_indices_to_run

        self.current_problem_global_idx: Optional[int] = None
        self.current_problem_path: Optional[str] = None
        self.current_problem_name: Optional[str] = None
        self.ref_arch_src: Optional[str] = None
        
        self.current_kernel_src: Optional[str] = None
        self.optimization_history: List[Dict[str, Any]] = []
        self.current_step_in_episode = 0

        self.ref_eager_time_ms = -1.0
        self.ref_compile_time_ms = -1.0 # torch.compile reference
        
        # For gymnasium.Env compliance, define action_space and observation_space
        # These are conceptual for text. Actual implementation might use tokenizers.
        # self.action_space = spaces.Text(max_length=1024, charset=string.printable) 
        # self.observation_space = spaces.Dict({
        #     "problem_id": spaces.Text(max_length=256),
        #     "problem_source": spaces.Text(max_length=15000), # Max length depends on dataset
        #     "current_kernel_source": spaces.Text(max_length=30000), # Can grow
        #     "optimization_history": spaces.Text(max_length=10000) # Summary of history
        # })


    def _select_next_problem(self):
        self._problem_iterator_idx = (self._problem_iterator_idx + 1) % len(self.problem_indices_to_run)
        self.current_problem_global_idx = self.problem_indices_to_run[self._problem_iterator_idx]
        
        self.current_problem_path = self.dataset[self.current_problem_global_idx]
        self.current_problem_name = os.path.basename(self.current_problem_path).replace(".py", "")
        self.ref_arch_src = read_file(self.current_problem_path)
        if not self.ref_arch_src:
            raise ValueError(f"Failed to read problem file: {self.current_problem_path}")

    def _get_reference_times(self, ref_src: str, problem_name_for_log: str) -> Tuple[float, float]:
        print(f"Calculating reference times for problem: {problem_name_for_log}...")
        eager_time, compile_time = -1.0, -1.0
        common_args = {
            "ref_arch_name": problem_name_for_log,
            "ref_arch_src": ref_src,
            "num_trials": self.num_perf_trials,
            "device": self.device,
            "verbose": False 
        }
        try:
            eager_stats = measure_program_time(**common_args, use_torch_compile=False)
            eager_time = eager_stats.get("mean", -1.0)

            if self.device.type == 'cuda': # torch.compile might not work well or be relevant on CPU
                compile_stats = measure_program_time(
                    **common_args,
                    use_torch_compile=True,
                    torch_compile_backend="inductor",
                    torch_compile_options="default"
                )
                compile_time = compile_stats.get("mean", -1.0)
            
            print(f"Reference times for {problem_name_for_log}: Eager={eager_time:.2f}ms, Compile={compile_time:.2f}ms")
        except Exception as e:
            print(f"Error calculating reference times for {problem_name_for_log}: {e}")
        return eager_time, compile_time

    def _get_observation(self) -> Dict[str, Any]:
        history_strings = []
        for item in self.optimization_history:
            eval_summary = "Eval: N/A"
            if item.get('eval_result'): # eval_result might be None if coder failed
                er = item['eval_result']
                eval_summary = f"Compiled={er.compiled}, Correct={er.correctness}, Runtime={er.runtime:.2f}ms"
            history_strings.append(
                f"Suggestion: '{item['suggestion']}'. {eval_summary}. Reward={item['reward']:.3f}"
            )
        
        return {
            "problem_id": self.current_problem_name,
            "problem_source": self.ref_arch_src,
            "current_kernel_source": self.current_kernel_src,
            "optimization_history": history_strings # List of strings
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed) 
        self._select_next_problem()
        self.current_kernel_src = self.ref_arch_src
        self.optimization_history = []
        self.current_step_in_episode = 0

        self.ref_eager_time_ms, self.ref_compile_time_ms = self._get_reference_times(self.ref_arch_src, self.current_problem_name)

        print(f"Reset to problem: {self.current_problem_name} (Level {self.kernel_bench_level})")
        obs = self._get_observation()
        info = {"ref_eager_time_ms": self.ref_eager_time_ms, "ref_compile_time_ms": self.ref_compile_time_ms}
        return obs, info

    def _evaluate_kernel(self, kernel_to_evaluate_src: str, kernel_id_str: str) -> KernelExecResult:
        kernel_hash = hashlib.md5(kernel_to_evaluate_src.encode()).hexdigest()
        build_dir = os.path.join(self.kernel_build_dir_base, 
                                 f"level{self.kernel_bench_level}",
                                 self.current_problem_name, 
                                 f"{kernel_id_str}_{kernel_hash}")
        
        if os.path.exists(build_dir): # Clean specific build_dir for a fresh compile
            shutil.rmtree(build_dir)
        os.makedirs(build_dir, exist_ok=True)

        eval_result = None
        try:
            eval_result = eval_kernel_against_ref(
                original_model_src=self.ref_arch_src,
                custom_model_src=kernel_to_evaluate_src,
                measure_performance=True, 
                verbose=False, # Make configurable if needed for debugging
                num_correct_trials=self.num_correct_trials,
                num_perf_trials=self.num_perf_trials,
                build_dir=build_dir,
                device=self.device
            )
        except Exception as e:
            print(f"Unhandled exception during kernel evaluation for {self.current_problem_name}/{kernel_id_str}: {e}")
            eval_result = KernelExecResult(compiled=False, correctness=False, metadata={"evaluation_exception": str(e)})
        
        # Ensure eval_result is not None
        return eval_result if eval_result is not None else KernelExecResult(compiled=False, correctness=False, metadata={"evaluation_internal_error": "eval_result was None"})


    def _calculate_reward(self, current_eval_result: KernelExecResult) -> float:
        reward = 0.0
        if not current_eval_result.compiled:
            return self.penalty_non_compilation

        if not current_eval_result.correctness:
            return self.penalty_incorrect
            
        reward += self.correctness_reward_weight 

        # Use eager time for speedup, as per run_and_check.py and common practice.
        # Potentially, could use min(ref_eager, ref_compile) or make configurable.
        baseline_time_for_speedup = self.ref_eager_time_ms 
        
        if baseline_time_for_speedup > 0 and current_eval_result.runtime > 0:
            speedup = baseline_time_for_speedup / current_eval_result.runtime
            reward += speedup 
        elif current_eval_result.runtime == -1.0: 
            # This means performance was not measured. If correct, this is unexpected.
             print(f"Warning: Correct kernel for {self.current_problem_name} but performance not measured (runtime is -1.0).")
        # else: other edge cases like baseline_time_for_speedup <= 0

        return reward

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self.ref_arch_src is None: # Should have been set by reset()
             raise RuntimeError("Environment not reset properly. Call reset() before step().")

        info = {}
        new_kernel_src = None
        eval_result = None

        try:
            # 1. Apply Action (GPT-4o implements the suggestion)
            print(f"Step {self.current_step_in_episode}: Problem '{self.current_problem_name}'. Applying suggestion: '{action[:100]}...'")
            new_kernel_src = self.gpt4o_coder_fn(
                problem_ref_src=self.ref_arch_src, # Pass original problem context
                current_kernel_src=self.current_kernel_src,
                suggestion=action
            )
            # 2. Evaluate Results
            print(f"Step {self.current_step_in_episode}: Evaluating new kernel...")
            eval_result = self._evaluate_kernel(new_kernel_src, f"step{self.current_step_in_episode}")
        
        except Exception as e: # Catch errors from coder_fn or _evaluate_kernel if they are not handled internally
            print(f"Critical error in step execution (coder or eval): {e}")
            # Create a dummy eval result for failure
            eval_result = KernelExecResult(compiled=False, correctness=False, metadata={"step_exception": str(e)})
            info["step_exception"] = str(e)
            # If new_kernel_src wasn't assigned due to coder error, use current_kernel_src for history
            if new_kernel_src is None: new_kernel_src = self.current_kernel_src


        # 3. Calculate Reward
        reward = self._calculate_reward(eval_result)
        print(f"Step {self.current_step_in_episode}: Eval: Compiled={eval_result.compiled}, Correct={eval_result.correctness}, Perf={eval_result.runtime:.2f}ms. Reward={reward:.3f}")
        
        # Update state
        self.current_kernel_src = new_kernel_src 
        self.optimization_history.append({
            "suggestion": action,
            "generated_kernel_src": new_kernel_src, # Storing the source for review
            "eval_result": eval_result, # Storing the object
            "reward": reward
        })
        
        self.current_step_in_episode += 1

        # 4. Determine `terminated` and `truncated`
        terminated = (self.current_step_in_episode >= self.max_steps_per_episode)
        if terminated:
            print(f"Episode terminated for {self.current_problem_name}: Max steps ({self.max_steps_per_episode}) reached.")
        
        # Optional: terminate early on severe failure
        # if not eval_result.compiled:
        #     terminated = True
        #     print(f"Episode terminated early for {self.current_problem_name} due to non-compilation.")

        truncated = False # Not used for now

        obs = self._get_observation()
        # Populate info dictionary more comprehensively
        info.update({
            "eval_result_compiled": eval_result.compiled,
            "eval_result_correctness": eval_result.correctness,
            "eval_result_runtime_ms": eval_result.runtime,
            "eval_result_metadata": eval_result.metadata, # Can be large
            "ref_eager_time_ms": self.ref_eager_time_ms,
            "ref_compile_time_ms": self.ref_compile_time_ms,
        })
        
        return obs, reward, terminated, truncated, info

    def render(self):
        pass # Not a visual environment

    def close(self):
        print("Closing KernelBenchRLEnv.")
        # Optional: Clean up self.kernel_build_dir_base if desired,
        # but typically caches are kept unless explicitly cleared.
        # if os.path.exists(self.kernel_build_dir_base):
        #     shutil.rmtree(self.kernel_build_dir_base)
        #     print(f"Cleaned up cache directory: {self.kernel_build_dir_base}")

# Example Usage (for testing the environment wrapper itself)
if __name__ == '__main__':
    print("Testing KernelBenchRLEnv...")

    # ---
    # Mock GPT-4o Coder: This is CRITICAL. 
    # For a real run, this must call your GPT-4o logic.
    # This mock simply returns the suggestion as if it's the new kernel code,
    # OR it can try to apply a very simple modification.
    # ---
    def mock_gpt4o_coder(problem_ref_src: str, current_kernel_src: str, suggestion: str) -> str:
        print(f"  [Mock GPT-4o Coder] Problem: ..., Current Kernel: ..., Suggestion: {suggestion}")
        if suggestion == "NO_OP":
            return current_kernel_src
        elif suggestion == "MAKE_IT_ADD_CONSTANT": # A simple, specific modification
            return current_kernel_src.replace("return result", "return result + 100.0 # Added constant by mock")
        elif suggestion.startswith("import torch"): # Assumes suggestion is full code
             return suggestion
        else: # Default: append as comment
            return current_kernel_src + f"\n# MOCK CODER: Applied suggestion '{suggestion}'\n"

    # Select a small subset of problems for quick testing if desired
    # Example: use first 2 problems from level 1
    # For level 1, problem indices are 0-based for the dataset list
    # test_problem_indices = [0, 1] 

    env = KernelBenchRLEnv(
        kernel_bench_level=1,
        gpt4o_coder_fn=mock_gpt4o_coder,
        # problems_subset_indices=test_problem_indices, # Optional: test on a small subset
        max_steps_per_episode=3,
        gpu_arch=["Ada"] if torch.cuda.is_available() else [], # Set your GPU Arch if CUDA is available
        device_id=0,
        clear_cache_on_start=True # Fresh start for test
    )

    num_episodes = 2 # Test a couple of problems
    for i_episode in range(num_episodes):
        print(f"\n--- Episode {i_episode + 1} ---")
        observation, info = env.reset()
        print(f"Initial Observation Keys: {observation.keys()}")
        print(f"Initial Problem ID: {observation['problem_id']}")
        # print(f"Initial Problem Source:\n{observation['problem_source'][:300]}...") # Can be very long
        
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0
        while not terminated and not truncated:
            step_count +=1
            # Action: In a real scenario, Qwen model generates this.
            # Here, we provide some mock suggestions.
            if step_count == 1:
                action = "NO_OP" # First action: do nothing
            elif step_count == 2 and "add" in observation['problem_id'].lower(): # Example of contextual action
                 action = read_file(os.path.join(REPO_TOP_DIR, "src/prompts/model_new_ex_add.py")) # Try a known good kernel
            elif step_count == 2 :
                 action = "TRY_KERNEL_FUSION_ABC" # A generic textual suggestion
            else:
                action = "MAKE_IT_ADD_CONSTANT"
            
            print(f"\n  Episode {i_episode+1}, Step {env.current_step_in_episode+1}, Action: {action[:50]}...")
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"  Observation (current_kernel_source snippet):\n{observation['current_kernel_source'][:200]}...")
            print(f"  Reward: {reward:.3f}")
            print(f"  Terminated: {terminated}, Truncated: {truncated}")
            if info:
                 print(f"  Info: compiled={info.get('eval_result_compiled')}, correct={info.get('eval_result_correctness')}, runtime={info.get('eval_result_runtime_ms', -1):.2f}ms")

        print(f"Episode {i_episode + 1} finished. Total reward: {total_reward:.3f}")

    env.close()
    print("\nKernelBenchRLEnv test finished.")