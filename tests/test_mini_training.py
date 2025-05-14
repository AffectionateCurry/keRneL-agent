import subprocess
import sys

def test_mini_training():
    """Run training with minimal settings"""
    cmd = [
        sys.executable, "src/train.py",
        "--model_name", "microsoft/phi-2",  # Small model
        "--level", "1",
        "--max_steps_train", "2",  # Just 2 steps
        "--grpo_collect_batch_size", "2",
        "--grpo_train_mini_batch_size", "1",
        "--save_steps", "1",
        "--output_dir", "./test_training_output",
        "--logging_dir", "./test_training_logs",
        "--max_steps_per_episode", "2",
        "--qwen_max_new_tokens", "64",  # Shorter outputs
        "--num_trajectories_for_buffer", "2"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)
        return result.returncode == 0
    except Exception as e:
        print(f"Training test failed: {e}")
        return False

if __name__ == "__main__":
    test_mini_training()