from ..kernelbench.src.eval import eval_kernel_against_ref
from ..kernelbench.src.utils import read_file
import torch

def test_simple_kernel_eval():
    # Use the example from prompts
    ref_kernel = read_file("kernelbench/src/prompts/model_ex_add.py")
    custom_kernel = read_file("kernelbench/src/prompts/model_new_ex_add.py")
    
    try:
        result = eval_kernel_against_ref(
            original_model_src=ref_kernel,
            custom_model_src=custom_kernel,
            num_correct_trials=1,
            num_perf_trials=5,
            verbose=True,
            measure_performance=True,
            device=torch.device("cuda:0")
        )
        print(f"Compilation: {result.compiled}")
        print(f"Correctness: {result.correctness}")
        print(f"Runtime: {result.runtime}ms")
        return result
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None

if __name__ == "__main__":
    test_simple_kernel_eval()