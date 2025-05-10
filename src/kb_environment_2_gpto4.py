import gym
from gym import spaces
import random
import os
import torch
import openai
from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.utils import extract_first_code

class KernelBenchEnv(gym.Env):
    """
    Gym environment for iterative kernel optimization with GRPO.

    Observation: 3-tuple of (previous-previous kernel A, previous kernel B, last suggestion).
    Action: suggestion string from the Qwen agent.
    Reward: 0.3 for correctness + speedup bonus (baseline_mean / new_runtime).
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        level: int,
        openai_api_key: str,
        suggestion_model_name: str = 'Qwen2.5-Coder-7B-Instruct',
        blackbox_model_name: str = 'gpt-4o',
        max_steps: int = 5,
        device: torch.device = None
    ):
        super().__init__()
        self.level = level
        self.dataset = construct_kernelbench_dataset(level)
        self.openai_api_key = openai_api_key
        self.suggestion_model_name = suggestion_model_name
        self.blackbox_model_name = blackbox_model_name
        self.max_steps = max_steps
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        # Observations: previous-previous code, previous code, last suggestion
        self.observation_space = spaces.Dict({
            'code_A': spaces.Text(),
            'code_B': spaces.Text(),
            'last_suggestion': spaces.Text(),
        })
        self.action_space = spaces.Text()

        self.reset()

    def reset(self):
        # Sample a random problem
        problem_path = random.choice(self.dataset)
        self.current_problem = problem_path
        with open(problem_path, 'r') as f:
            self.original_src = f.read()

        # Compute baseline runtime (original vs. original)
        baseline_res = eval_kernel_against_ref(
            original_model_src=self.original_src,
            custom_model_src=self.original_src,
            measure_performance=True,
            num_correct_trials=1,
            num_perf_trials=10,
            build_dir=None,
            device=self.device
        )
        self.baseline_mean = baseline_res.runtime_stats.get('mean', None)

        # Initialize code history
        self.code_A = self.original_src
        self.code_B = self.original_src
        self.last_suggestion = ""
        self.step_count = 0
        return self._get_obs()

    def step(self, action: str):
        # Record suggestion
        self.last_suggestion = action

        # Apply suggestion via blackbox LLM (GPT-4o)
        openai.api_key = self.openai_api_key
        prompt = f"""
You are a code optimization assistant. Given the following kernels and a suggestion, apply the suggestion and return the updated kernel code only.

Kernel A:
```python
{self.code_A}
```

Kernel B:
```python
{self.code_B}
```

Suggestion:
{action}

Output only the updated Python code in a code block.
"""
        response = openai.ChatCompletion.create(
            model=self.blackbox_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for code optimization."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.0,
            max_tokens=4096
        )
        content = response.choices[0].message.content
        code_C = extract_first_code(content, ["python"]) or self.code_B

        # Evaluate new kernel
        result = eval_kernel_against_ref(
            original_model_src=self.original_src,
            custom_model_src=code_C,
            measure_performance=True,
            num_correct_trials=1,
            num_perf_trials=10,
            build_dir=None,
            device=self.device
        )

        # Compute reward
        reward = 0.0
        if result.compiled and result.correctness:
            reward += 0.3
            mean = result.runtime_stats.get('mean')
            if self.baseline_mean and mean:
                reward += self.baseline_mean / mean

        # Episode done?
        self.step_count += 1
        done = result.correctness or (self.step_count >= self.max_steps)

        # Shift history: B->A, C->B
        self.code_A = self.code_B
        self.code_B = code_C

        obs = self._get_obs()
        info = {'result': result}
        return obs, reward, done, info

    def _get_obs(self):
        return {
            'code_A': self.code_A,
            'code_B': self.code_B,
            'last_suggestion': self.last_suggestion,
        }

    def render(self, mode='human'):
        print(f"Problem: {os.path.basename(self.current_problem)}")
        print("--- Kernel A ---")
        print(self.code_A[:500] + '...' if len(self.code_A) > 500 else self.code_A)
        print("--- Kernel B ---")
        print(self.code_B[:500] + '...' if len(self.code_B) > 500 else self.code_B)
        print("Last suggestion:", self.last_suggestion)

    def close(self):
        pass
