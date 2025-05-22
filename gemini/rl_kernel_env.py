import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import os
import random
from typing import List, Tuple, Dict, Any, Callable, Optional
import hashlib # Added import
import logging

# Corrected imports assuming kernelbench is installed and /src is part of PYTHONPATH
from kernelbench.src.dataset import construct_kernelbench_dataset, KERNEL_BENCH_PATH
from kernelbench.src.eval import eval_kernel_against_ref, KernelExecResult
# get_timing_stats is used by measure_program_time, not directly here.
from kernelbench.src.utils import set_gpu_arch, read_file
# Importing from scripts is not ideal but necessary if kernelbench is not modified.
# This requires kernelbench directory to be in PYTHONPATH for scripts to be found.
from kernelbench.scripts.generate_baseline_time import measure_program_time

from .coder import gpt4o_code_generator # Correct relative import

logger = logging.getLogger(__name__)

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
                 gpt_coder_fn: Callable[[str, str, str, int, float], str] = gpt4o_code_generator, # Blackbox LLM
                 blackbox_llm_model_name: str = "gpt-4o",
                 max_steps_per_episode: int = 4, # 4 refinement steps
                 gpu_arch_list: List[str] = ["Ada"], 
                 device_id: int = 0, # Use -1 for CPU explicitly
                 num_correct_trials_eval: int = 3, # Reduced for faster iteration during RL
                 num_perf_trials_eval: int = 30,  # Reduced for faster iteration
                 correctness_reward: float = 0.3,
                 penalty_non_compilation: float = -1.0,
                 penalty_incorrect: float = -0.5,
                 problem_subset_indices: Optional[List[int]] = None,
                 build_cache_dir_base: Optional[str] = None,
                 verbose_eval: bool = False
                 ):
        super().__init__()

        self.kernel_bench_level = kernel_bench_level
        self.gpt_coder_fn = gpt_coder_fn
        self.blackbox_llm_model_name = blackbox_llm_model_name
        self.max_steps_per_episode = max_steps_per_episode
        
        self.num_correct_trials_eval = num_correct_trials_eval
        self.num_perf_trials_eval = num_perf_trials_eval
        self.correctness_reward_val = correctness_reward
        self.penalty_non_compilation = penalty_non_compilation
        self.penalty_incorrect = penalty_incorrect
        self.verbose_eval = verbose_eval

        if device_id >= 0 and torch.cuda.is_available():
            if device_id >= torch.cuda.device_count():
                logger.warning(f"CUDA device_id {device_id} out of range. Found {torch.cuda.device_count()} devices. Using cuda:0.")
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device(f"cuda:{device_id}")
            set_gpu_arch(gpu_arch_list) # This sets an environment variable
            self.gpu_arch = gpu_arch_list
            logger.info(f"KernelBenchRLEnv using device: {self.device} with arch {self.gpu_arch}")
        else:
            logger.warning("CUDA not available or device_id < 0. Running on CPU. Kernel evaluation may fail or be very slow.")
            self.device = torch.device("cpu")
            self.gpu_arch = []

        self.dataset_paths = construct_kernelbench_dataset(level=self.kernel_bench_level)
        if not self.dataset_paths:
            raise ValueError(f"Could not load KernelBench dataset for level {self.kernel_bench_level} from {KERNEL_BENCH_PATH}")

        if problem_subset_indices:
            self.problem_indices_to_run = [idx for idx in problem_subset_indices if 0 <= idx < len(self.dataset_paths)]
            if not self.problem_indices_to_run:
                raise ValueError("Problem subset indices are out of bounds or result in an empty set.")
        else:
            self.problem_indices_to_run = list(range(len(self.dataset_paths)))
        
        if not self.problem_indices_to_run:
             raise ValueError("No problems to run. Check dataset path and subset indices.")
        self._problem_iterator_idx = -1 

        self.action_space = spaces.Text(min_length=1, max_length=512) # Max length of Qwen suggestion
        self.observation_space = spaces.Dict({
            "code_A_src": spaces.Text(max_length=65536), # Kernel before B
            "code_B_src": spaces.Text(max_length=65536), # Current kernel to optimize
            "last_suggestion_A_to_B": spaces.Text(max_length=512) # Suggestion that led to B
        })
        
        # Ensure KERNEL_BENCH_PATH is valid if used in build_cache_dir_base default
        default_cache_path = "/tmp/rl_kb_cache" # More generic default
        if KERNEL_BENCH_PATH and os.path.exists(KERNEL_BENCH_PATH):
             default_cache_path = os.path.join(KERNEL_BENCH_PATH, ".rl_cache")

        self.build_cache_dir_base = build_cache_dir_base or default_cache_path
        try:
            os.makedirs(self.build_cache_dir_base, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create cache directory {self.build_cache_dir_base}: {e}. Using /tmp/rl_kb_cache_fallback")
            self.build_cache_dir_base = "/tmp/rl_kb_cache_fallback"
            os.makedirs(self.build_cache_dir_base, exist_ok=True)


        self.original_ref_src: Optional[str] = None
        self.current_problem_path: Optional[str] = None
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
        
        try:
            self.original_ref_src = read_file(self.current_problem_path)
            if not self.original_ref_src:
                logger.error(f"Failed to read problem file (empty content): {self.current_problem_path}")
                # Skip to next problem or raise error
                raise ValueError(f"Empty problem file: {self.current_problem_path}")
        except Exception as e:
            logger.error(f"Exception reading problem file {self.current_problem_path}: {e}")
            raise # Re-raise to signal a critical issue

    def _get_reference_time(self, ref_src: str, problem_name_for_log: str) -> float:
        logger.info(f"Calculating reference time for problem: {problem_name_for_log}...")
        if self.device.type == 'cpu':
            logger.warning("Skipping reference time calculation on CPU.")
            return -1.0
        try:
            stats = measure_program_time(
                ref_arch_name=f"{problem_name_for_log}_ref_eager",
                ref_arch_src=ref_src,
                num_trials=self.num_perf_trials_eval,
                use_torch_compile=False, # Eager baseline
                device=self.device,
                verbose=self.verbose_eval 
            )
            eager_time_ms = stats.get("mean", -1.0)
            if eager_time_ms == -1.0 and "error" in stats.get("metadata", {}): # measure_program_time might not populate metadata like this
                 logger.error(f"Reference time calculation failed for {problem_name_for_log}. Error: {stats['metadata']['error']}")
            elif eager_time_ms > 0 :
                 logger.info(f"Reference eager time for {problem_name_for_log}: {eager_time_ms:.2f}ms")
            else:
                 logger.warning(f"Reference eager time for {problem_name_for_log} is not positive: {eager_time_ms:.2f}ms. Stats: {stats}")
            return eager_time_ms
        except Exception as e:
            logger.error(f"Error calculating reference time for {problem_name_for_log}: {e}", exc_info=True)
            return -1.0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        self._select_next_problem()

        assert self.original_ref_src is not None, "original_ref_src not set after _select_next_problem"
        self.code_A_src = self.original_ref_src 
        self.code_B_src = self.original_ref_src 
        self.last_suggestion_A_to_B = "" 
        self.current_step_in_episode = 0
        
        self.baseline_eager_time_ms = self._get_reference_time(self.original_ref_src, self.current_problem_name or "unknown_problem")

        logger.info(f"Reset to problem: {self.current_problem_name} (Level {self.kernel_bench_level})")
        obs = self._get_observation()
        info = {"baseline_eager_time_ms": self.baseline_eager_time_ms, "problem_name": self.current_problem_name}
        return obs, info

    def _get_observation(self) -> Dict[str, Any]:
        return {
            "code_A_src": str(self.code_A_src or ""),
            "code_B_src": str(self.code_B_src or ""),
            "last_suggestion_A_to_B": str(self.last_suggestion_A_to_B or "")
        }

    def _evaluate_kernel(self, kernel_to_evaluate_src: str, step_info: str) -> KernelExecResult:
        if self.device.type == 'cpu':
            logger.warning(f"Skipping kernel evaluation on CPU for {self.current_problem_name}/{step_info}.")
            return KernelExecResult(compiled=False, correctness=False, metadata={"skipped_on_cpu": True})

        # Create a unique build directory for each evaluation attempt.
        # This avoids issues with stale cache or concurrent builds if this env were parallelized (though not typical for a single env instance).
        kernel_hash = hashlib.md5(kernel_to_evaluate_src.encode()).hexdigest()
        problem_name_safe = "".join(c if c.isalnum() else "_" for c in (self.current_problem_name or "unknown"))
        
        unique_build_dir = os.path.join(
            self.build_cache_dir_base,
            f"level{self.kernel_bench_level}",
            problem_name_safe,
            f"{step_info}_{kernel_hash}"
        )
        
        eval_result = None
        try:
            logger.info(f"Evaluating kernel for {self.current_problem_name}/{step_info}. Build dir: {unique_build_dir}")
            eval_result = eval_kernel_against_ref(
                original_model_src=self.original_ref_src,
                custom_model_src=kernel_to_evaluate_src,
                measure_performance=True, 
                verbose= True,
                num_correct_trials=self.num_correct_trials_eval,
                num_perf_trials=self.num_perf_trials_eval,
                build_dir=unique_build_dir,
                device=self.device
            )
        except Exception as e:
            logger.error(f"Unhandled exception during kernel evaluation for {self.current_problem_name}/{step_info}: {e}", exc_info=True)
            eval_result = KernelExecResult(compiled=False, correctness=False, metadata={"evaluation_exception": str(e)})
        
        return eval_result if eval_result is not None else KernelExecResult(compiled=False, correctness=False, metadata={"evaluation_internal_error": "eval_result was None"})


    def _calculate_reward(self, eval_result: KernelExecResult) -> float:
        if not eval_result.compiled:
            return self.penalty_non_compilation 
        if not eval_result.correctness:
            return self.penalty_incorrect

        reward = self.correctness_reward_val 

        if self.baseline_eager_time_ms > 0 and eval_result.runtime is not None and eval_result.runtime > 0:
            # Speedup reward: KernelBench typically uses (baseline_time / new_time).
            # A speedup > 1 is good.
            # We can scale this. For example, (speedup - 1.0) so no improvement is 0 reward.
            # Or log speedup: log(baseline / new_time) = log(baseline) - log(new_time)
            # Let's use a simple scaled speedup for now.
            speedup = self.baseline_eager_time_ms / eval_result.runtime
            # Cap reward to avoid extreme values from very fast (potentially unstable) kernels
            # Example: Max speedup reward of 2.0 (for 3x speedup if base reward is 1 for 1x)
            speedup_reward = min(max(speedup - 1.0, -0.5), 2.0) # Penalize if slower, cap positive
            reward += speedup_reward
            logger.info(f"Speedup: {speedup:.2f}x (Baseline: {self.baseline_eager_time_ms:.2f}ms, New: {eval_result.runtime:.2f}ms). Speedup reward component: {speedup_reward:.3f}")
        elif eval_result.runtime == -1.0 and eval_result.correctness:
            logger.warning(f"Correct kernel for {self.current_problem_name} but perf not measured (runtime is -1.0). No speedup reward.")
        
        return reward

    def step(self, action_qwen_suggestion: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self.original_ref_src is None or self.code_B_src is None:
            logger.error("Environment not reset properly. Call reset() before step().")
            # Return a dummy observation and terminate to prevent further errors
            dummy_obs = self._get_observation() # Will use None if state vars are None
            return dummy_obs, 0.0, True, False, {"error": "Environment not reset"}

        self.current_step_in_episode += 1
        info: Dict[str, Any] = {}
        
        logger.info(f"Step {self.current_step_in_episode}/{self.max_steps_per_episode} for problem '{self.current_problem_name}'. Applying suggestion: '{action_qwen_suggestion[:100]}...'")
        
        new_kernel_C_src = self.gpt_coder_fn(
            current_kernel_src=self.code_B_src, # type: ignore
            suggestion=action_qwen_suggestion,
            blackbox_model_name=self.blackbox_llm_model_name,
            # max_tokens and temperature are defaulted in gpt_coder_fn
        )

        logger.info(f"Step {self.current_step_in_episode}: Evaluating new kernel C...")
        eval_result = self._evaluate_kernel(new_kernel_C_src, f"step{self.current_step_in_episode}")
        
        reward = self._calculate_reward(eval_result)
        logger.info(f"Step {self.current_step_in_episode}: Eval Result: Compiled={eval_result.compiled}, Correct={eval_result.correctness}, Perf={eval_result.runtime:.2f}ms (if measured). Calculated Reward={reward:.3f}")

        # Update state for next observation
        self.code_A_src = self.code_B_src
        self.code_B_src = new_kernel_C_src
        self.last_suggestion_A_to_B = action_qwen_suggestion
        
        terminated = (self.current_step_in_episode >= self.max_steps_per_episode)
        if terminated:
            logger.info(f"Episode terminated for {self.current_problem_name}: Max steps ({self.max_steps_per_episode}) reached.")
        
        truncated = False 
        
        obs = self._get_observation()
        info.update({
            "eval_result_compiled": eval_result.compiled,
            "eval_result_correctness": eval_result.correctness,
            "eval_result_runtime_ms": eval_result.runtime,
            "eval_result_metadata": eval_result.metadata, # Already serializable from kernelbench
            "baseline_eager_time_ms": self.baseline_eager_time_ms,
            "problem_name": self.current_problem_name,
            "step_in_episode": self.current_step_in_episode,
        })
        
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        print(f"\n--- Render ---")
        print(f"Problem: {self.current_problem_name} (Level {self.kernel_bench_level})")
        print(f"Current Step in Episode: {self.current_step_in_episode}/{self.max_steps_per_episode}")
        print(f"Baseline Eager Time: {self.baseline_eager_time_ms:.2f}ms")
        obs = self._get_observation()
        print(f"Last Suggestion (A->B): {obs['last_suggestion_A_to_B'][:200]}...")
        print(f"Kernel B (current for Qwen's input): \n{obs['code_B_src'][:300]}...")
        # print(f"Kernel A (previous for Qwen's input): \n{obs['code_A_src'][:300]}...")
        print(f"--- End Render ---\n")

    def close(self):
        logger.info("Closing KernelBenchRLEnv.")