import torch
from typing import Dict, List, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
import modal
from claude.rl_kernel_env import KernelBenchRLEnv


class KernelBenchGRPOEnv:
    """GRPO-compatible wrapper for KernelBench RL environment with Modal support."""
    
    def __init__(
        self,
        rl_env_config: Dict[str, Any],
        qwen_system_prompt: str = "You are an expert CUDA kernel optimization assistant.",
        qwen_prompt_template: str = """Previous kernel (A):
```python
{kernel_a_src}
Previous suggestion that transformed A to B:
"{last_suggestion}"
Current kernel (B) to optimize:
{kernel_b_src}
Provide a concise optimization suggestion for kernel B:""",
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.env = KernelBenchRLEnv(**rl_env_config)
        self.qwen_system_prompt = qwen_system_prompt
        self.qwen_prompt_template = qwen_prompt_template
        self.generation_kwargs = generation_kwargs or {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        }

    def _create_prompt(self, observation: Dict[str, Any]) -> str:
        """Create Qwen prompt from observation."""
        return self.qwen_prompt_template.format(
            kernel_a_src=observation["kernel_a_src"],
            kernel_b_src=observation["kernel_b_src"],
            last_suggestion=observation["last_suggestion"]
        )

    @modal.method()
    def generate_trajectory(
        self,
        agent_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ) -> Dict[str, List[Any]]:
        """Generate a complete trajectory for GRPO training."""
        queries = []
        responses = []
        rewards = []
        
        obs, info = self.env.reset()
        
        # Ensure proper padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if self.generation_kwargs.get("pad_token_id") is None:
            self.generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
        
        done = False
        while not done:
            # Create prompt for Qwen
            prompt = self._create_prompt(obs)
            queries.append(prompt)
            
            # Tokenize prompt
            if hasattr(tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "system", "content": self.qwen_system_prompt},
                    {"role": "user", "content": prompt}
                ]
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(agent_model.device)
            else:
                full_prompt = f"{self.qwen_system_prompt}\n\n{prompt}"
                input_ids = tokenizer.encode(
                    full_prompt,
                    return_tensors="pt"
                ).to(agent_model.device)
            
            # Generate suggestion
            with torch.no_grad():
                output_ids = agent_model.generate(
                    input_ids,
                    **self.generation_kwargs
                )
            
            # Decode suggestion
            generated_ids = output_ids[0][input_ids.shape[1]:]
            suggestion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            responses.append(suggestion)
            
            # Execute step
            next_obs, reward, terminated, truncated, info = self.env.step(suggestion)
            rewards.append(reward)
            
            obs = next_obs
            done = terminated or truncated
        
        return {
            "queries": queries,
            "responses": responses,
            "rewards": rewards,
            "problem_name": info.get("problem_name", "unknown"),
            "final_speedup": info.get("speedup", 0.0)
        }

    @modal.method()
    def batch_generate_trajectories(
        self,
        agent_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        num_trajectories: int,
    ) -> List[Dict[str, List[Any]]]:
        """Generate multiple trajectories for a batch."""
        trajectories = []
        
        for i in range(num_trajectories):
            print(f"Generating trajectory {i+1}/{num_trajectories}")
            try:
                trajectory = self.generate_trajectory(agent_model, tokenizer)
                trajectories.append(trajectory)
            except Exception as e:
                print(f"Error generating trajectory {i+1}: {e}")
                continue
        
        return trajectories