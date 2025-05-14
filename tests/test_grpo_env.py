from src.kernelbench_grpo_env import KernelBenchGRPOEnv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_grpo_env():
    # Use a small model for testing
    model_name = "microsoft/phi-2"  # Small model for testing
    
    # Minimal config
    rl_env_config = {
        "kernel_bench_level": 1,
        "max_steps_per_episode": 2,
        "gpu_arch_list": ["Ada"],
        "device_id": 0,
        "problem_subset_indices": [0],  # Single problem
        "num_correct_trials_eval": 1,
        "num_perf_trials_eval": 3,
        "build_cache_dir_base": "./test_cache"
    }
    
    generation_kwargs = {
        "max_new_tokens": 128,  # Shorter for testing
        "temperature": 0.7,
        "do_sample": True
    }
    
    try:
        # Initialize
        env = KernelBenchGRPOEnv(
            rl_env_config=rl_env_config,
            generation_kwargs=generation_kwargs
        )
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Generate trajectory
        print("Generating trajectory...")
        trajectory = env.generate_trajectory(model, tokenizer)
        
        print(f"Queries: {len(trajectory['queries'])}")
        print(f"Responses: {len(trajectory['responses'])}")
        print(f"Rewards: {trajectory['rewards']}")
        
        return trajectory
    except Exception as e:
        print(f"GRPO env test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_grpo_env()