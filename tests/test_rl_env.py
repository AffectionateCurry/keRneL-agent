# test_rl_env.py
from src.rl_kernel_env import KernelBenchRLEnv
import torch

def test_rl_env_basic():
    config = {
        "kernel_bench_level": 1,
        "max_steps_per_episode": 2,  # Short episodes for testing
        "gpu_arch_list": ["Ada"],
        "device_id": 0,
        "num_correct_trials_eval": 1,
        "num_perf_trials_eval": 5,
        "correctness_reward": 0.3,
        "problem_subset_indices": [0, 1],  # Only first two problems
        "build_cache_dir_base": "./test_cache"
    }
    
    try:
        env = KernelBenchRLEnv(**config)
        obs, info = env.reset()
        
        print("Initial observation keys:", obs.keys())
        print("Code A length:", len(obs["code_A_src"]))
        print("Code B length:", len(obs["code_B_src"]))
        print("Baseline time:", info.get("baseline_eager_time_ms"))
        
        # Test step
        test_suggestion = "Optimize memory access patterns"
        next_obs, reward, terminated, truncated, step_info = env.step(test_suggestion)
        
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print("Step info:", step_info)
        
        return True
    except Exception as e:
        print(f"RL env test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_rl_env_basic()