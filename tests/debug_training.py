import logging
import torch
from src.train import main
from argparse import Namespace

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add hooks to debug
original_step = None

def debug_step_wrapper(self, action):
    """Wrapper to debug step function"""
    print(f"\n=== DEBUG STEP ===")
    print(f"Action type: {type(action)}")
    print(f"Action content: {action[:100]}...")
    
    result = original_step(action)
    
    obs, reward, terminated, truncated, info = result
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Info keys: {info.keys()}")
    print("==================\n")
    
    return result

# Monkey patch for debugging
from src import rl_kernel_env
original_step = rl_kernel_env.KernelBenchRLEnv.step
rl_kernel_env.KernelBenchRLEnv.step = debug_step_wrapper

# Run with debug settings
debug_args = Namespace(
    model_name="microsoft/phi-2",
    gpt_model="gpt-3.5-turbo",  # Cheaper for testing
    level=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir="./debug_output",
    logging_dir="./debug_logs",
    max_steps_per_episode=1,
    qwen_max_prompt_len=512,
    qwen_max_new_tokens=64,
    qwen_temperature=0.7,
    qwen_top_p=0.9,
    grpo_collect_batch_size=1,
    grpo_train_mini_batch_size=1,
    gradient_accumulation_steps=1,
    ppo_epochs=1,
    learning_rate=1e-5,
    gamma=0.4,
    max_steps_train=1,
    save_steps=1,
    deepspeed_config=None
)

if __name__ == "__main__":
    try:
        main(debug_args)
    except Exception as e:
        import traceback
        print(f"Error during training: {e}")
        traceback.print_exc()