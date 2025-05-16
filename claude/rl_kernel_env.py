import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import os
import hashlib
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import modal

from KernelBench.src.dataset import construct_kernelbench_dataset
from KernelBench.src.eval import eval_kernel_against_ref, KernelExecResult
from KernelBench.src.utils import set_gpu_arch, read_file
from KernelBench.scripts.generate_baseline_time import measure_program_time

from claude.coder import KernelCoder

class KernelBenchRLEnv(gym.Env):
    """RL Environment for KernelBench optimization with Modal GPU support."""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        kernel_bench_level: int,
        kernel_coder: Optional[KernelCoder] = None,
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
        modal_gpu_config: str = "a10g:1",
    ):
        super().__init__()
        
        self.level = kernel_bench_level
        self.kernel_coder = kernel_coder or KernelCoder()
        self.max_steps_per_episode = max_steps_per_episode
        self.gpu_arch_list = gpu_arch_list
        self.device_id = device_id
        self.num_correct_trials = num_correct_trials
        self.num_perf_trials = num_perf_trials
        
        # Rewards and penalties
        self.correctness_reward = correctness_reward
        self.compilation_penalty = compilation_penalty
        self.incorrectness_penalty = incorrectness_penalty
        
        # Setup cache directory
        self.cache_dir = Path(cache_dir or "/tmp/kernelbench_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
            set_gpu_arch(gpu_arch_list)
        else:
            print("WARNING: CUDA not available, using CPU")
            self.device = torch.device("cpu")
        
        # Load dataset
        self.dataset = construct_kernelbench_dataset(level=self.level)
        if not self.dataset:
            raise ValueError(f"Failed to load KernelBench level {self.level}")
        
        # Problem selection
        if problem_subset:
            self.problem_indices = [i for i in problem_subset if 0 <= i < len(self.dataset)]
        else:
            self.problem_indices = list(range(len(self.dataset)))
        
        if not self.problem_indices:
            raise ValueError("No valid problems selected")
        
        self.current_problem_idx = 0
        
        # Modal GPU configuration
        self.modal_gpu_config = modal_gpu_config
        
        # Define action/observation spaces
        self.action_space = spaces.Text(min_length=1, max_length=2048)
        self.observation_space = spaces.Dict({
            "kernel_a_src": spaces.Text(max_length=65536),
            "kernel_b_src": spaces.Text(max_length=65536),
            "last_suggestion": spaces.Text(max_length=2048)
        })
        
        # Episode state
        self._reset_episode_state()
    
    def _reset_episode_state(self):
        """Reset episode-specific state."""
        self.ref_src = None
        self.kernel_a_src = None
        self.kernel_b_src = None
        self.last_suggestion = ""
        self.step_count = 0
        self.baseline_time = -1.0
        self.problem_name = None
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        # Select next problem
        problem_idx = self.problem_indices[self.current_problem_idx]
        self.current_problem_idx = (self.current_problem_idx + 1) % len(self.problem_indices)
        
        # Load problem
        problem_path = self.dataset[problem_idx]
        self.problem_name = Path(problem_path).stem
        self.ref_src = read_file(problem_path)
        
        if not self.ref_src:
            raise ValueError(f"Failed to read problem: {problem_path}")
        
        # Initialize kernels
        self.kernel_a_src = self.ref_src
        self.kernel_b_src = self.ref_src
        self.last_suggestion = ""
        self.step_count = 0
        
        # Calculate baseline performance
        self.baseline_time = self._measure_baseline()
        
        print(f"Reset to problem: {self.problem_name} (Level {self.level})")
        
        obs = self._get_observation()
        info = {
            "problem_name": self.problem_name,
            "baseline_time_ms": self.baseline_time
        }
        
        return obs, info
    
    def step(self, action: str):
        """Execute one optimization step."""
        self.step_count += 1
        
        print(f"Step {self.step_count}: Applying suggestion: {action[:50]}...")
        
        # Generate new kernel using GPT-4o
        new_kernel = self.kernel_coder.generate_kernel(
            current_kernel_src=self.kernel_b_src,
            suggestion=action
        )
        
        # Evaluate new kernel
        eval_result = self._evaluate_kernel(new_kernel)
        reward = self._calculate_reward(eval_result)
        
        print(f"Step {self.step_count}: Compiled={eval_result.compiled}, "
              f"Correct={eval_result.correctness}, "
              f"Runtime={eval_result.runtime:.2f}ms, Reward={reward:.3f}")
        
        # Update state
        self.kernel_a_src = self.kernel_b_src
        self.kernel_b_src = new_kernel
        self.last_suggestion = action
        
        # Check termination
        terminated = self.step_count >= self.max_steps_per_episode
        truncated = False
        
        obs = self._get_observation()
        info = {
            "eval_result": eval_result.dict(),
            "baseline_time_ms": self.baseline_time,
            "speedup": self.baseline_time / eval_result.runtime if eval_result.runtime > 0 else 0
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, str]:
        """Get current observation."""
        return {
            "kernel_a_src": self.kernel_a_src,
            "kernel_b_src": self.kernel_b_src, 
            "last_suggestion": self.last_suggestion
        }
    
    def _measure_baseline(self) -> float:
        """Measure baseline performance of reference implementation."""
        if self.device.type == 'cpu':
            print("Skipping baseline measurement on CPU")
            return -1.0
        
        try:
            stats = measure_program_time(
                ref_arch_name=f"{self.problem_name}_baseline",
                ref_arch_src=self.ref_src,
                num_trials=self.num_perf_trials,
                use_torch_compile=False,
                device=self.device,
                verbose=False
            )
            baseline_time = stats.get("mean", -1.0)
            print(f"Baseline time: {baseline_time:.2f}ms")
            return baseline_time
        except Exception as e:
            print(f"Error measuring baseline: {e}")
            return -1.0
    
    def _evaluate_kernel(self, kernel_src: str) -> KernelExecResult:
        """Evaluate a kernel implementation."""
        # Create unique build directory
        kernel_hash = hashlib.md5(kernel_src.encode()).hexdigest()[:8]
        build_dir = self.cache_dir / f"level{self.level}" / self.problem_name / f"step{self.step_count}_{kernel_hash}"
        build_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            result = eval_kernel_against_ref(
                original_model_src=self.ref_src,
                custom_model_src=kernel_src,
                measure_performance=True,
                verbose=False,
                num_correct_trials=self.num_correct_trials,
                num_perf_trials=self.num_perf_trials,
                build_dir=str(build_dir),
                device=self.device
            )
            return result
        except Exception as e:
            print(f"Evaluation error: {e}")
            return KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={"error": str(e)}
            )
    
    def _calculate_reward(self, eval_result: KernelExecResult) -> float:
        """Calculate reward based on evaluation results."""
        if not eval_result.compiled:
            return self.compilation_penalty
        
        if not eval_result.correctness:
            return self.incorrectness_penalty
        
        # Base reward for correctness
        reward = self.correctness_reward
        
        # Add speedup reward
        if self.baseline_time > 0 and eval_result.runtime > 0:
            speedup = self.baseline_time / eval_result.runtime
            reward += speedup
        
        return reward
    
    def render(self, mode='human'):
        """Render current state."""
        print(f"\n--- Episode Step {self.step_count} ---")
        print(f"Problem: {self.problem_name}")
        print(f"Baseline Time: {self.baseline_time:.2f}ms")
        print(f"Last Suggestion: {self.last_suggestion}")
        print("---\n")