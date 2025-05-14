import torch
from ..kernelbench.src.dataset import construct_kernelbench_dataset

def test_cuda_available():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    return torch.cuda.is_available()

def test_dataset_loading():
    try:
        dataset_level1 = construct_kernelbench_dataset(level=1)
        print(f"Level 1 problems: {len(dataset_level1)}")
        print(f"First problem: {dataset_level1[0]}")
        return True
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        return False

if __name__ == "__main__":
    test_cuda_available()
    test_dataset_loading()