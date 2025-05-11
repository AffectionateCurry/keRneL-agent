import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import os
import random
from typing import List, Tuple, Dict, Any, Callable, Optional

from src.dataset import construct_kernelbench_dataset, KERNEL_BENCH_PATH
from ..kernelbench.src.eval import eval_kernel_against_ref, KernelExecResult, get_timing_stats
from ..kernelbench.src.utils import set_gpu_arch, read_file
from ..kernelbench.scripts.generate_baseline_time import measure_program_time # For baseline timing
from coder import gpt4o_code_generator # Import the blackbox coder

class KernelBenchRLEnv(gym.Env):
    """
    An RL Environment Wrapper for KernelBench based on Gymnasium.
    Observation: (Kernel A src, Kernel B src, Suggestion that led A to B)
    Action: An optimization suggestion string from the Qwen model.
    Reward: Based on correctness and performance improvement.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self,
                 kernel_bench_level: int,
                 gpt_coder_fn: Callable[[str, str], str] = gpt4o_code_generator, # Blackbox LLM
                 blackbox_llm_model_name: str = "gpt-4o",
                 max_steps_per_episode: int = 4, # 4 refinement steps
                 gpu_arch_list: List[str] = ["Ada"], # Example, adjust to your hardware
                 device_id: int = 0,
                 num_correct_trials_eval: int = 5,
                 num_perf_trials_eval: int = 100,
                 correctness_reward: float = 0.3,
                 # Penalties from user prompt not explicitly mentioned for this env,
                 # but good to have. For now, focusing on positive reward.
                 # penalty_non_compilation: float = -1.0,
                 # penalty_incorrect: float = -0.5,
                 problem_subset_indices: Optional[List[int]] = None,
                 build_cache_dir_base: Optional[str] = None
                 ):
        super().__init__()

        self.kernel_bench_level = kernel_bench_level
        self.gpt_coder_fn = gpt_coder_fn
        self.blackbox_llm_model_name = blackbox_llm_model_name
        self.max_steps_per_episode = max_steps_per_episode
        
        self.num_correct_trials_eval = num_correct_trials_eval
        self.num_perf_trials_eval = num_perf_trials_eval
        self.correctness_reward_val = correctness_reward

        if torch.cuda.is_available():
            if device_id >= torch.cuda.device_count():
                print(f"Warning: CUDA device_id {device_id} out of range. Found {torch.cuda.device_count()} devices. Using cuda:0.")
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device(f"cuda:{device_id}")
            set_gpu_arch(gpu_arch_list)
            self.gpu_arch = gpu_arch_list
        else:
            print("Warning: CUDA not available. Running on CPU. Kernel evaluation will likely fail or be very slow.")
            self.device = torch.device("cpu")
            self.gpu_arch = []

        self.dataset_paths = construct_kernelbench_dataset(level=self.kernel_bench_level)
        if not self.dataset_paths:
            raise ValueError(f"Could not load KernelBench dataset for level {self.kernel_bench_level}")

        if problem_subset_indices:
            self.problem_indices_to_run = [idx for idx in problem_subset_indices if 0 <= idx < len(self.dataset_paths)]
            if not self.problem_indices_to_run:
                raise ValueError("Problem subset indices are out of bounds or result in an empty set.")
        else:
            self.problem_indices_to_run = list(range(len(self.dataset_paths)))
        
        self._problem_iterator_idx = -1 # Iterates through self.problem_indices_to_run

        # Define action and observation spaces (conceptual for text)
        # Max length can be tuned based on typical kernel/suggestion sizes
        self.action_space = spaces.Text(min_length=1, max_length=2048) 
        self.observation_space = spaces.Dict({
            "code_A_src": spaces.Text(max_length=65536),
            "code_B_src": spaces.Text(max_length=65536),
            "last_suggestion_A_to_B": spaces.Text(max_length=2048)
        })

        self.build_cache_dir_base = build_cache_dir_base or os.path.join(KERNEL_BENCH_PATH, ".rl_cache")
        os.makedirs(self.build_cache_dir_base, exist_ok=True)

        # State variables, initialized in reset
        self.original_ref_src: Optional[str] = None
        self.current_problem_name: Optional[str] = None
        self.code_A_src: Optional[str] = None
        self.code_B_src: Optional[str] = None
        self.last_suggestion_A_to_B: Optional[str] = None
        self.current_step_in_episode: int = 0
        self.baseline_eager_time_ms: float = -1.0

    def _select_next_problem(self):
        self._problem_iterator_idx = (self._problem_iterator_idx + 1) % len(self.problem_indices_to_run)
        problem_dataset_idx = self.problem_indices_to_run[self._problem_iterator_idx]
        
        self.current_problem_path = self.dataset_paths[problem_dataset_idx]
        self.current_problem_name = os.path.basename(self.current_problem_path).replace(".py", "")
        self.original_ref_src = read_file(self.current_problem_path)
        if not self.original_ref_src:
            raise ValueError(f"Failed to read problem file: {self.current_problem_path}")

    def _get_reference_time(self, ref_src: str, problem_name_for_log: str) -> float:
        print(f"Calculating reference time for problem: {problem_name_for_log}...")
        if self.device.type == 'cpu':
            print("Warning: Skipping reference time calculation on CPU.")
            return -1.0
        try:
            # We use eager time as the primary baseline for speedup, as per KernelBench practices
            stats = measure_program_time(
                ref_arch_name=f"{problem_name_for_log}_ref_eager",
                ref_arch_src=ref_src,
                num_trials=self.num_perf_trials_eval, # Use eval trials for consistency
                use_torch_compile=False,
                device=self.device,
                verbose=False
            )
            eager_time_ms = stats.get("mean", -1.0)
            print(f"Reference eager time for {problem_name_for_log}: {eager_time_ms:.2f}ms")
            return eager_time_ms
        except Exception as e:
            print(f"Error calculating reference time for {problem_name_for_log}: {e}")
            return -1.0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        self._select_next_problem()

        assert self.original_ref_src is not None
        self.code_A_src = self.original_ref_src
        self.code_B_src = self.original_ref_src
        self.last_suggestion_A_to_B = "" # Initial state: no prior suggestion
        self.current_step_in_episode = 0
        
        self.baseline_eager_time_ms = self._get_reference_time(self.original_ref_src, self.current_problem_name) # type: ignore

        print(f"Reset to problem: {self.current_problem_name} (Level {self.kernel_bench_level})")
        obs = self._get_observation()
        info = {"baseline_eager_time_ms": self.baseline_eager_time_ms}
        return obs, info

    def _get_observation(self) -> Dict[str, Any]:
        # Ensure all components are strings, even if None initially (though reset should handle this)
        return {
            "code_A_src": str(self.code_A_src or ""),
            "code_B_src": str(self.code_B_src or ""),
            "last_suggestion_A_to_B": str(self.last_suggestion_A_to_B or "")
        }

    def _evaluate_kernel(self, kernel_to_evaluate_src: str, step_info: str) -> KernelExecResult:
        unique_build_dir = os.path.join(
            self.build_cache_dir_base,
            f"level{self.kernel_bench_level}",
            self.current_problem_name, # type: ignore
            f"{step_info}_{hashlib.md5(kernel_to_evaluate_src.encode()).hexdigest()}"
        )
        # os.makedirs(unique_build_dir, exist_ok=True) # eval_kernel_against_ref handles this

        eval_result = None
        try:
            eval_result = eval_kernel_against_ref(
                original_model_src=self.original_ref_src, # type: ignore
                custom_model_src=kernel_to_evaluate_src,
                measure_performance=True, # Always measure for reward
                verbose=False, # Can be made configurable
                num_correct_trials=self.num_correct_trials_eval,
                num_perf_trials=self.num_perf_trials_eval,
                build_dir=unique_build_dir,
                device=self.device
            )
        except Exception as e:
            print(f"Unhandled exception during kernel evaluation for {self.current_problem_name}/{step_info}: {e}")
            eval_result = KernelExecResult(compiled=False, correctness=False, metadata={"evaluation_exception": str(e)})
        
        return eval_result if eval_result is not None else KernelExecResult(compiled=False, correctness=False, metadata={"evaluation_internal_error": "eval_result was None"})

    def _calculate_reward(self, eval_result: KernelExecResult) -> float:
        reward = 0.0
        if not eval_result.compiled:
            return -1.0 # Default penalty for non-compilation
        if not eval_result.correctness:
            return -0.5 # Default penalty for incorrectness (but compiled)

        reward += self.correctness_reward_val # 0.3 for correctness

        if self.baseline_eager_time_ms > 0 and eval_result.runtime is not None and eval_result.runtime > 0:
            speedup = self.baseline_eager_time_ms / eval_result.runtime
            reward += speedup
        elif eval_result.runtime == -1.0: # Performance not measured, though correct
            print(f"Warning: Correct kernel for {self.current_problem_name} but perf not measured (runtime is -1.0). No speedup reward.")
        
        return reward

    def step(self, action_qwen_suggestion: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self.original_ref_src is None or self.code_B_src is None:
            raise RuntimeError("Environment not reset properly. Call reset() before step().")

        self.current_step_in_episode += 1
        info: Dict[str, Any] = {}
        
        print(f"Step {self.current_step_in_episode}: Problem '{self.current_problem_name}'. Applying suggestion: '{action_qwen_suggestion[:100]}...'")
        
        # 1. Apply Action (Blackbox LLM implements the suggestion)
        # self.gpt_coder_fn takes current_kernel_src (self.code_B_src) and suggestion
        new_kernel_C_src = self.gpt_coder_fn(
            current_kernel_src=self.code_B_src,
            suggestion=action_qwen_suggestion,
            # blackbox_model_name=self.blackbox_llm_model_name # If gpt_coder_fn needs it
        )

        # 2. Evaluate Results
        print(f"Step {self.current_step_in_episode}: Evaluating new kernel...")
        eval_result = self._evaluate_kernel(new_kernel_C_src, f"step{self.current_step_in_episode}")
        
        # 3. Calculate Reward
        reward = self._calculate_reward(eval_result)
        print(f"Step {self.current_step_in_episode}: Eval: Compiled={eval_result.compiled}, Correct={eval_result.correctness}, Perf={eval_result.runtime:.2f}ms. Reward={reward:.3f}")

        # 4. Update state
        self.code_A_src = self.code_B_src
        self.code_B_src = new_kernel_C_src
        self.last_suggestion_A_to_B = action_qwen_suggestion
        
        # 5. Determine `terminated` and `truncated`
        terminated = (self.current_step_in_episode >= self.max_steps_per_episode)
        if terminated:
            print(f"Episode terminated for {self.current_problem_name}: Max steps ({self.max_steps_per_episode}) reached.")
        
        # Early termination if kernel is really bad (e.g., doesn't compile)
        # This can be tuned. For now, we continue to max_steps.
        # if not eval_result.compiled:
        #     terminated = True
        #     print(f"Episode terminated early for {self.current_problem_name} due to non-compilation.")

        truncated = False # Not used for now
        
        obs = self._get_observation()
        info.update({
            "eval_result_compiled": eval_result.compiled,
            "eval_result_correctness": eval_result.correctness,
            "eval_result_runtime_ms": eval_result.runtime,
            "eval_result_metadata": eval_result.metadata,
            "baseline_eager_time_ms": self.baseline_eager_time_ms,
        })
        
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        print(f"--- Episode Step {self.current_step_in_episode} ---")
        print(f"Problem: {self.current_problem_name}")
        print(f"Baseline Eager Time: {self.baseline_eager_time_ms:.2f}ms")
        print("--- Kernel A (Previous-Previous) ---")
        print(f"{str(self.code_A_src)[:500]}...")
        print("--- Kernel B (Previous) ---")
        print(f"{str(self.code_B_src)[:500]}...")
        print(f"--- Last Suggestion (A->B) ---")
        print(str(self.last_suggestion_A_to_B))
        print("------------------------------------")

    def close(self):
        print("Closing KernelBenchRLEnv.")