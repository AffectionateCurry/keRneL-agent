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
            input_ids = None
            attention_mask = None

            if hasattr(tokenizer, 'apply_chat_template'):
                # ---- CORRECTED BLOCK ----
                messages = [
                    {"role": "system", "content": self.qwen_system_prompt},
                    {"role": "user", "content": prompt}
                ]
                try:
                    tokenized_inputs = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True, # Important for instruction-following models
                        return_tensors="pt",
                        return_attention_mask=True # Ensure attention_mask is returned
                    )
                    input_ids = tokenized_inputs.input_ids.to(agent_model.device)
                    # apply_chat_template might not return attention_mask if not explicitly handled
                    # by the tokenizer's template or if return_attention_mask isn't effective for it.
                    # It's safer to get it if present, otherwise fall back to creating one.
                    if "attention_mask" in tokenized_inputs:
                        attention_mask = tokenized_inputs.attention_mask.to(agent_model.device)
                    else:
                        # If apply_chat_template doesn't provide it, but we have input_ids,
                        # we can create a default one (all ones).
                        # This assumes no padding *within* the templated prompt itself,
                        # which is usually the case for generation prompts.
                        print("Warning: tokenizer.apply_chat_template did not explicitly return an attention_mask. Creating a default one.")
                        attention_mask = torch.ones_like(input_ids)

                except Exception as e:
                    print(f"Error using apply_chat_template: {e}. Falling back to manual concatenation.")
                    # Fallback to manual concatenation if apply_chat_template fails or is problematic
                    full_prompt = f"{self.qwen_system_prompt}\n\n{prompt}"
                    tokenized_output = tokenizer.encode_plus(
                        full_prompt,
                        return_tensors="pt",
                        return_attention_mask=True,
                        truncation=True, # Good practice to add truncation
                        max_length=tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 2048 # or a sensible default
                    )
                    input_ids = tokenized_output.input_ids.to(agent_model.device)
                    attention_mask = tokenized_output.attention_mask.to(agent_model.device)
            else: # Fallback if apply_chat_template is not available
                full_prompt = f"{self.qwen_system_prompt}\n\n{prompt}"
                tokenized_output = tokenizer.encode_plus(
                    full_prompt,
                    return_tensors="pt",
                    return_attention_mask=True,
                    truncation=True,
                    max_length=tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 2048
                )
                input_ids = tokenized_output.input_ids.to(agent_model.device)
                attention_mask = tokenized_output.attention_mask.to(agent_model.device)
            # ---- END CORRECTED BLOCK ----

            if input_ids is None or attention_mask is None:
                # This should not happen if tokenization is successful
                print("Error: Tokenization failed to produce input_ids or attention_mask. Skipping trajectory.")
                # Potentially break or return an empty/error trajectory
                return { 
                    "queries": [], "responses": [], "rewards": [], 
                    "problem_name": "tokenization_error", "final_speedup": 0.0
                }

            # Generate suggestion
            with torch.no_grad():
                output_ids = agent_model.generate(
                    input_ids,
                    attention_mask=attention_mask, # <--- PASS ATTENTION MASK HERE
                    **self.generation_kwargs
                )

            # Decode suggestion
            # The generated IDs start after the input_ids
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