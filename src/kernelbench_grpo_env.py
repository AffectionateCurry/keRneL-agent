from typing import Dict, List, Any, Callable, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from rl_kernel_env import KernelBenchRLEnv
import logging

logger = logging.getLogger(__name__)

class KernelBenchGRPOEnv:
    """
    A wrapper environment for TRL's GRPOTrainer, using KernelBenchRLEnv internally.
    This class is responsible for generating full trajectories.
    """
    def __init__(self,
                 rl_env_config: Dict[str, Any], # Config for KernelBenchRLEnv
                 # Qwen prompt generation details
                 qwen_system_prompt: str = "You are an expert CUDA kernel optimization assistant.",
                 qwen_prompt_template: str = """Kernel A (previous-previous version code):
```python
{code_A_src}

Suggestion that transformed Kernel A to Kernel B:
"{last_suggestion_A_to_B}"

Kernel B (current kernel code to optimize):

{code_B_src}
Based on the above, provide your concise and actionable optimization suggestion for Kernel B, which will be implemented by a separate blackbox code generation LLM. Output only the suggestion text, without any preamble or code blocks.
Your suggestion:""",
        generation_kwargs: Optional[Dict[str, Any]] = None
        ):
        self.env = KernelBenchRLEnv(**rl_env_config)
        self.qwen_system_prompt = qwen_system_prompt
        self.qwen_prompt_template = qwen_prompt_template
        self.generation_kwargs = generation_kwargs or {
            "max_new_tokens": 256, # Max length of Qwen's suggestion
            "pad_token_id": self.env.device # Dummy, will be set by tokenizer if None
        }
        if self.generation_kwargs.get("pad_token_id") is self.env.device : # Check if it's the dummy
            self.generation_kwargs["pad_token_id"] = None # Will be set by tokenizer

    def _create_qwen_prompt(self, observation: Dict[str, Any]) -> str:
        # The observation from KernelBenchRLEnv contains code_A_src, code_B_src, last_suggestion_A_to_B
        return self.qwen_prompt_template.format(**observation)

    def generate_trajectory(self,
                            agent_model: PreTrainedModel,
                            tokenizer: PreTrainedTokenizer
                        ) -> Dict[str, List[Any]]:
        """
        Generates a single trajectory (episode) using the agent model.
        """
        trajectory_obs = []
        trajectory_actions_str = [] # Qwen's suggestions (strings)
        trajectory_action_tokens = [] # Tokenized Qwen's suggestions
        trajectory_rewards = []
        
        obs, info = self.env.reset()
        
        if self.generation_kwargs.get("pad_token_id") is None and tokenizer.pad_token_id is not None:
            self.generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
        elif self.generation_kwargs.get("pad_token_id") is None and tokenizer.eos_token_id is not None:
            self.generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
            logger.warning(f"Using EOS token ({tokenizer.eos_token_id}) as PAD token for generation.")


        for step in range(self.env.max_steps_per_episode):
            full_prompt_for_qwen = self._create_qwen_prompt(obs)
            
            # Tokenize prompt for Qwen
            # Use chat template if model expects it, otherwise direct tokenization
            if hasattr(tokenizer, 'apply_chat_template') and isinstance(obs, dict): # Basic check
                # This assumes the prompt template is for a user turn
                messages = [
                    {"role": "system", "content": self.qwen_system_prompt},
                    {"role": "user", "content": full_prompt_for_qwen}
                ]
                tokenized_prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(agent_model.device)
            else:
                # Fallback for simpler models or if system prompt is part of main prompt
                tokenized_prompt = tokenizer.encode(full_prompt_for_qwen, return_tensors="pt").to(agent_model.device)

            trajectory_obs.append(full_prompt_for_qwen) # Store the string prompt
            
            # Generate action (suggestion) using Qwen
            with torch.no_grad():
                action_tokens = agent_model.generate(tokenized_prompt, **self.generation_kwargs)
            
            # Decode Qwen's suggestion
            # action_str = tokenizer.decode(action_tokens[0], skip_special_tokens=True)
            # Only decode the generated part
            generated_tokens = action_tokens[0][tokenized_prompt.shape[1]:]
            action_str = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            trajectory_actions_str.append(action_str)
            trajectory_action_tokens.append(action_tokens[0].cpu()) # Store full output tokens for TRL

            # Step the environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_str)
            
            trajectory_rewards.append(reward)
            
            obs = next_obs
            if terminated or truncated:
                break
        
        # GRPOTrainer expects 'queries' (prompts given to Qwen), 'responses' (Qwen's generated suggestions)
        # and 'rewards'.
        return {
            "queries": trajectory_obs, # List of string prompts to Qwen
            "responses": trajectory_actions_str, # List of string suggestions from Qwen
            "rewards": trajectory_rewards, # List of scalar rewards
            # For more direct use with GRPOTrainer if it prefers tokenized versions for loss calculation:
            # "query_tokens": [tokenizer.encode(q, return_tensors="pt") for q in trajectory_obs],
            # "response_tokens": [tokenizer.encode(r, return_tensors="pt") for r in trajectory_actions_str],
        }

    def get_final_results(self) -> Dict[str, Any]: # Helper if needed after evaluation episodes
        # This could return aggregated stats from the last episode if rl_kernel_env stores them.
        # For now, this is conceptual.
        return {"info": "Final results placeholder from GRPO Env wrapper."}

