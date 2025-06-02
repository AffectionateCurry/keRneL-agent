# claude/rl_kernel_env.py
# --- BEGIN MODIFICATIONS ---
import sys
from pathlib import Path

# 1) Locate key directories
_current_file_dir  = Path(__file__).parent.resolve()   # .../keRneL-agent/claude
_project_root      = _current_file_dir.parent         # .../keRneL-agent
_kernelbench_dir   = _project_root / "KernelBench"    # .../keRneL-agent/KernelBench

# 2) Insert KernelBench itself first so that `import src.*` → KernelBench/src
if str(_kernelbench_dir) not in sys.path:
    sys.path.insert(0, str(_kernelbench_dir))

# 3) Then insert your project root so that fully‐qualified imports still work:
#    - import KernelBench.src.*
#    - import claude.*
if str(_project_root) not in sys.path:
    sys.path.insert(1, str(_project_root))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import os
import hashlib
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import traceback
import modal

from KernelBench.src.dataset import construct_kernelbench_dataset
from KernelBench.src.eval import eval_kernel_against_ref, KernelExecResult

# print("eval_kernel_against_ref is →", eval_kernel_against_ref) # Keep for debugging if needed
assert callable(eval_kernel_against_ref), (
    "❌ eval_kernel_against_ref isn’t a function! Your import is still wrong."
)

from KernelBench.src.utils import set_gpu_arch, read_file, extract_first_code # Added extract_first_code
from KernelBench.scripts.generate_baseline_time import measure_program_time
# Removed: from claude.coder import KernelCoder 

class KernelBenchRLEnv(gym.Env):
    """RL Environment for KernelBench optimization. Qwen generates full code."""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        kernel_bench_level: int,
        # kernel_coder: Optional[KernelCoder] = None, # REMOVED: Qwen will generate code
        max_steps_per_episode: int = 4,
        gpu_arch_list: List[str] = ["Ada"],
        device_id: int = 0,
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        correctness_reward: float = 0.3,
        compilation_penalty: float = -1.0,
        incorrectness_penalty: float = -0.5,
        problem_subset: Optional[List[int]] = None,
        cache_dir: Optional[str] = None,
        modal_gpu_config: str = "a10g:1", # This seems more for Modal config, less for core env logic
    ):
        super().__init__()
        
        self.level = kernel_bench_level
        # self.kernel_coder = kernel_coder # REMOVED
        self.max_steps_per_episode = max_steps_per_episode
        self.gpu_arch_list = gpu_arch_list
        self.device_id = device_id
        self.num_correct_trials = num_correct_trials
        self.num_perf_trials = num_perf_trials
        
        self.correctness_reward = correctness_reward
        self.compilation_penalty = compilation_penalty
        self.incorrectness_penalty = incorrectness_penalty
        
        self.cache_dir = Path(cache_dir or "/tmp/kernelbench_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
            set_gpu_arch(gpu_arch_list)
        else:
            print("WARNING: CUDA not available, using CPU")
            self.device = torch.device("cpu")
        
        self.dataset = construct_kernelbench_dataset(level=self.level)
        if not self.dataset:
            raise ValueError(f"Failed to load KernelBench level {self.level}")
        
        if problem_subset:
            self.problem_indices = [i for i in problem_subset if 0 <= i < len(self.dataset)]
        else:
            self.problem_indices = list(range(len(self.dataset)))
        
        if not self.problem_indices:
            raise ValueError("No valid problems selected")
        
        self.current_problem_idx = 0 # Index within self.problem_indices
        
        # Action is now the full Python/CUDA module string
        self.action_space = spaces.Text(min_length=10, max_length=131072) # Increased max_length significantly
        self.observation_space = spaces.Dict({
            "kernel_a_src": spaces.Text(max_length=131072), # Increased
            "kernel_b_src": spaces.Text(max_length=131072), # Increased
            "last_generated_code_info": spaces.Text(max_length=2048) # Was last_suggestion
        })
        
        self._reset_episode_state()
    
    def _reset_episode_state(self):
        self.ref_src = None
        self.kernel_a_src = None  # Stores the source of the kernel *before* the last action
        self.kernel_b_src = None  # Stores the source of the kernel *after* the last action (current state)
        self.last_generated_code_info = "" # Info about the last code Qwen generated
        self.step_count = 0
        self.baseline_time = -1.0
        self.problem_name = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
        problem_idx: Optional[int] = None, 
    ):
        super().reset(seed=seed)
        
        selected_problem_list_idx: int
        if problem_idx is None: # Usual round-robin based on self.current_problem_idx
            selected_problem_list_idx = self.current_problem_idx
            self.current_problem_idx = (self.current_problem_idx + 1) % len(self.problem_indices)
        else: # Fixed by caller (problem_idx is the actual index in self.dataset)
            try:
                # Find its position in our (potentially subset) problem_indices list
                # This is mostly for internal tracking if needed, but caller provides the direct dataset index
                selected_problem_list_idx = self.problem_indices.index(problem_idx)
            except ValueError:
                # If problem_idx is not in our subset, this is an issue.
                # However, GRPOEnv will likely pass an index from self.problem_indices
                assert problem_idx in self.problem_indices, f"problem_idx {problem_idx} not in environment's problem_indices list."
                # If problem_idx is a direct index into self.dataset, use it.
                # For consistency, the problem_idx passed should be an index from self.dataset

        # Ensure problem_idx (which is an index for self.dataset) is valid
        actual_dataset_idx = problem_idx if problem_idx is not None else self.problem_indices[selected_problem_list_idx]
        assert 0 <= actual_dataset_idx < len(self.dataset), f"Invalid problem_idx for dataset: {actual_dataset_idx}"

        problem_path = self.dataset[actual_dataset_idx]
        self.problem_name = Path(problem_path).stem
        self.ref_src = read_file(problem_path)
        
        if not self.ref_src:
            raise ValueError(f"Failed to read problem: {problem_path}")
        
        self.kernel_a_src = self.ref_src # Before any optimization
        self.kernel_b_src = self.ref_src # Current state to be optimized
        self.last_generated_code_info = "Initial state, no code generated yet."
        self.step_count = 0
        
        self.baseline_time = self._measure_baseline()
        
        print(f"Reset to problem: {self.problem_name} (Dataset index {actual_dataset_idx}, Level {self.level})")
        
        obs = self._get_observation()
        info = {
            "problem_name": self.problem_name,
            "baseline_time_ms": self.baseline_time,
            "kernel_b_src": self.kernel_b_src, # For GRPO wrapper to have initial state
        }
        
        return obs, info
    
    def step(self, action_code: str): # Action is now the full code string
        self.step_count += 1
        
        # The 'action_code' is the new kernel code generated by Qwen
        new_kernel_src = action_code
        
        print(f"Step {self.step_count}: Evaluating new kernel code (length {len(new_kernel_src)})...")
        
        # Validate if new_kernel_src is a 'ModelNew' string etc.
        # For now, assume it's what eval_kernel_against_ref expects
        if not new_kernel_src or not ("class ModelNew(nn.Module):" in new_kernel_src):
            print(f"Warning: Received potentially malformed or empty kernel code at step {self.step_count}.")
            # Handle malformed code: assign penalty, use old kernel, etc.
            # For now, let eval_kernel_against_ref try and likely fail compilation.
            eval_result = KernelExecResult(
                compiled=False,
                correctness=False,
                runtime=-1,
                metadata={"error": "Malformed or empty action_code received by environment."}
            )
        else:
            eval_result = self._evaluate_kernel(new_kernel_src)

        reward = self._calculate_reward(eval_result)
        
        print(f"Step {self.step_count}: Compiled={eval_result.compiled}, "
              f"Correct={eval_result.correctness}, "
              f"Runtime={eval_result.runtime:.2f}ms, Reward={reward:.3f}")
        
        # Update state: kernel_a becomes what kernel_b was, kernel_b becomes new_kernel_src
        self.kernel_a_src = self.kernel_b_src
        if eval_result.compiled and eval_result.correctness: # Only update to new kernel if it's good
            self.kernel_b_src = new_kernel_src
        else:
            # If new kernel is bad, kernel_b_src effectively remains the same as kernel_a_src
            # (or rather, the state before this failed attempt).
            # The prompt to Qwen should still use the kernel_b_src that was *attempted* to be optimized.
            # Let's ensure kernel_b_src reflects the *outcome* of the step.
            # If eval fails, we don't want kernel_b to become the bad code for the *next* step's input.
            # So, kernel_b_src should only be updated if the new code is valid.
            # This means the *next* observation's kernel_b_src will be the same as this step's kernel_a_src
            # if the current `new_kernel_src` was bad.
            # However, for constructing the *next* prompt, Qwen needs to know about kernel_b (the one it tried to optimize)
            # and the `new_kernel_src` (the one it generated that failed).
            # The `obs` should reflect the state Qwen works on.
            # Let's stick to: kernel_b_src IS UPDATED if new_kernel_src compiled and was correct.
            # If not, kernel_b_src REMAINS what it was at the START of this step.
            # This means self.kernel_a_src = self.kernel_b_src (done above)
            # and if bad: self.kernel_b_src = self.kernel_a_src (effectively reverting)
            # This implies the next observation's kernel_b_src is the one *before* this failed attempt.
            # This seems correct for learning. Qwen should try to optimize the *last known good state*.
            # The prompt will still contain kernel_a (state before last attempt) and kernel_b (state it tried to optimize FROM).
            pass # kernel_b_src is NOT updated with new_kernel_src if it's bad

        self.last_generated_code_info = f"Attempted code (len {len(new_kernel_src)}), Result: C={eval_result.compiled}, OK={eval_result.correctness}, R={eval_result.runtime:.2f}ms"
        
        terminated = self.step_count >= self.max_steps_per_episode
        truncated = False # Could add other truncation conditions
        
        obs = self._get_observation() # This will now reflect the updated kernel_a and kernel_b
        info = {
            "eval_result": eval_result.dict(),
            "baseline_time_ms": self.baseline_time,
            "speedup": (self.baseline_time / eval_result.runtime) if (self.baseline_time > 0 and eval_result.runtime > 0 and eval_result.correctness) else 0.0,
            "kernel_b_src": self.kernel_b_src, # Current state of kernel_b after the step
            "problem_name": self.problem_name,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, str]:
        return {
            "kernel_a_src": self.kernel_a_src or "N/A (initial state)",
            "kernel_b_src": self.kernel_b_src or "N/A (initial state)", 
            "last_generated_code_info": self.last_generated_code_info
        }
    
    def _measure_baseline(self) -> float:
        if self.device.type == 'cpu':
            print("Skipping baseline measurement on CPU")
            return -1.0
        
        try:
            print(f"Measuring baseline for {self.problem_name} on {self.device}...")
            stats = measure_program_time(
                ref_arch_name=f"{self.problem_name}_baseline",
                ref_arch_src=self.ref_src, # This is the original problem source
                num_trials=self.num_perf_trials,
                use_torch_compile=False,
                device=self.device,
                verbose=False # Reduce verbosity
            )
            baseline_time = stats.get("mean", -1.0)
            if baseline_time > 0:
                print(f"Baseline time for {self.problem_name}: {baseline_time:.2f}ms")
            else:
                print(f"Failed to measure baseline time for {self.problem_name} (result: {baseline_time}).")
            return baseline_time
        except Exception as e:
            print(f"Error measuring baseline for {self.problem_name}: {e}")
            traceback.print_exc()
            return -1.0
    
    def _evaluate_kernel(self, kernel_src: str) -> KernelExecResult:
        kernel_hash = hashlib.md5(kernel_src.encode()).hexdigest()[:8]
        build_dir = self.cache_dir / f"level{self.level}" / self.problem_name / f"step{self.step_count}_{kernel_hash}"
        build_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # original_model_src should be the problem's reference src for comparison
            result = eval_kernel_against_ref(
                original_model_src=self.ref_src, 
                custom_model_src=kernel_src, # This is the Qwen-generated ModelNew code
                measure_performance=True,
                verbose=True, # Reduce verbosity; can be True for deep debugging
                num_correct_trials=self.num_correct_trials,
                num_perf_trials=self.num_perf_trials,
                build_dir=str(build_dir),
                device=self.device
            )
            return result
        except Exception as e:
            traceback.print_exc()
            print(f"Evaluation error during eval_kernel_against_ref: {e}")
            return KernelExecResult(
                compiled=False,
                correctness=False,
                runtime=-1,
                metadata={"error": str(e)}
            )
    
    def _calculate_reward(self, eval_result: KernelExecResult) -> float:
        if not eval_result.compiled:
            return self.compilation_penalty
        
        if not eval_result.correctness:
            return self.incorrectness_penalty
        
        reward = self.correctness_reward
        
        if self.baseline_time > 0 and eval_result.runtime > 0:
            speedup = self.baseline_time / eval_result.runtime
            # Cap reward to prevent extreme values, e.g. reward += min(speedup, 10.0)
            # Linear speedup reward for now
            reward += speedup 
        elif self.baseline_time <= 0 and eval_result.runtime > 0 : # No baseline, but runs
             reward += 0.1 # Small reward for running if no baseline
        
        return reward
    
    def render(self, mode='human'):
        print(f"\n--- Env State: Problem {self.problem_name}, Step {self.step_count} ---")
        print(f"Baseline Time: {self.baseline_time:.2f}ms")
        print(f"Last Generated Code Info: {self.last_generated_code_info}")
        print(f"Kernel A (before last attempt) hash: {hashlib.md5(self.kernel_a_src.encode()).hexdigest()[:8] if self.kernel_a_src else 'N/A'}")
        print(f"Kernel B (current state to optimize) hash: {hashlib.md5(self.kernel_b_src.encode()).hexdigest()[:8] if self.kernel_b_src else 'N/A'}")
        print("---\n")

# --- END MODIFICATIONS ---