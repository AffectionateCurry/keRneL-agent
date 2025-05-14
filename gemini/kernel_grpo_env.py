# File: /src/kernelbench_grpo_env.py
from typing import Dict, List, Any, Callable, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from .rl_kernel_env import KernelBenchRLEnv  # Correct relative import
import logging

logger = logging.getLogger(__name__)


class KernelBenchGRPOEnv:
    """
    A wrapper environment for TRL's GRPOTrainer, using KernelBenchRLEnv internally.
    This class is responsible for generating full trajectories.
    """
    def __init__(
        self,
        rl_env_config: Dict[str, Any],
        qwen_system_prompt: str = "You are an expert CUDA kernel optimization assistant.",
        qwen_prompt_template: str = """Given the previous kernel (Kernel A), the suggestion that led to the current kernel (Kernel B), and Kernel B itself, your task is to propose a new, concise, and actionable optimization suggestion for Kernel B. This suggestion will be implemented by a separate advanced code generation model. Focus on a single, impactful optimization.

Kernel A (previous version):
```python
{code_A_src}
Use code with caution.
Python
Suggestion that transformed Kernel A to Kernel B:
"{last_suggestion_A_to_B}"
Kernel B (current kernel code to optimize):
{code_B_src}
Use code with caution.
Python
Based on the above, provide your concise and actionable optimization suggestion for Kernel B. Output only the suggestion text, without any preamble, explanations, or code blocks.
Your suggestion:""",
        generation_kwargs: Optional[Dict[str, Any]] = None
    ):
        logger.info(f"Initializing KernelBenchGRPOEnv with RL Env config: {rl_env_config}")
        self.env = KernelBenchRLEnv(**rl_env_config)
        self.qwen_system_prompt = qwen_system_prompt
        self.qwen_prompt_template = qwen_prompt_template

        # Default generation_kwargs, allow pad_token_id to be set later by tokenizer
        self.generation_kwargs = generation_kwargs or {}
        self.generation_kwargs.setdefault("max_new_tokens", 256)  # Max length of Qwen's suggestion
        # pad_token_id will be set in generate_trajectory if not already in self.generation_kwargs
        logger.info(f"KernelBenchGRPOEnv generation_kwargs (initial): {self.generation_kwargs}")

    def _create_qwen_prompt(self, observation: Dict[str, Any]) -> str:
        # Ensure keys exist in observation, provide empty string as fallback
        # This should ideally not happen if KernelBenchRLEnv.reset() is correct
        code_A = observation.get("code_A_src", "# Code A not available")
        code_B = observation.get("code_B_src", "# Code B not available")
        last_suggestion = observation.get("last_suggestion_A_to_B", "# No previous suggestion")

        return self.qwen_prompt_template.format(
            code_A_src=code_A,
            code_B_src=code_B,
            last_suggestion_A_to_B=last_suggestion
        )

    def generate_trajectory(
        self,
        agent_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer
    ) -> Dict[str, List[Any]]:
        """
        Generates a single trajectory (episode) using the agent model.
        """
        trajectory_obs_prompts_str = []
        trajectory_actions_suggestions_str = []
        trajectory_rewards = []

        obs, info = self.env.reset()  # obs is a dict from _get_observation()
        problem_name = info.get("problem_name", "unknown_problem")
        logger.info(f"Starting trajectory for problem: {problem_name}")

        # Set pad_token_id in generation_kwargs if not already set
        if self.generation_kwargs.get("pad_token_id") is None:
            if tokenizer.pad_token_id is not None:
                self.generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
            elif tokenizer.eos_token_id is not None:
                self.generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
                logger.warning(f"Using EOS token ({tokenizer.eos_token_id}) as PAD token for Qwen generation.")
            else:
                logger.error("Cannot determine pad_token_id for Qwen generation. Tokenizer has no pad_token or eos_token.")
                # Potentially raise an error or use a default if absolutely necessary
                # For now, model.generate might handle it or error out.

        for step_num in range(self.env.max_steps_per_episode):
            full_prompt_for_qwen = self._create_qwen_prompt(obs)
            trajectory_obs_prompts_str.append(full_prompt_for_qwen)

            # Determine agent_model's device; default to self.env.device if model has no specific device
            model_device = agent_model.device if hasattr(agent_model, 'device') else self.env.device

            # Check for chat template
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
                messages = [
                    {"role": "system", "content": self.qwen_system_prompt},
                    {"role": "user", "content": full_prompt_for_qwen}
                ]
                tokenized_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model_device)
            else:
                tokenized_prompt = tokenizer.encode(
                    self.qwen_system_prompt + "\n" + full_prompt_for_qwen,
                    return_tensors="pt"
                ).to(model_device)

            logger.debug(f"Trajectory {problem_name} - Step {step_num+1}: Qwen prompt (first 100 chars): {full_prompt_for_qwen[:100]}...")

            with torch.no_grad():
                action_tokens = agent_model.generate(tokenized_prompt, **self.generation_kwargs)

            generated_tokens = action_tokens[0][tokenized_prompt.shape[1]:]
            action_suggestion_str = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            logger.debug(f"Trajectory {problem_name} - Step {step_num+1}: Qwen suggestion: {action_suggestion_str[:100]}...")
            trajectory_actions_suggestions_str.append(action_suggestion_str)

            next_obs, reward, terminated, truncated, step_info = self.env.step(action_suggestion_str)

            trajectory_rewards.append(reward)

            obs = next_obs
            if terminated or truncated:
                logger.info(f"Trajectory for problem {problem_name} ended at step {step_num+1}. Terminated: {terminated}, Truncated: {truncated}")
                break

        logger.info(f"Finished trajectory for problem: {problem_name}. Total steps: {len(trajectory_rewards)}. Rewards: {trajectory_rewards}")
        return {
            "queries": trajectory_obs_prompts_str,
            "responses": trajectory_actions_suggestions_str,
            "rewards": trajectory_rewards,
        }

    def get_environment_details(self) -> Dict[str, Any]:
        return {
            "max_steps_per_episode": self.env.max_steps_per_episode,
            "kernel_bench_level": self.env.kernel_bench_level,
            "observation_space": str(self.env.observation_space),
            "action_space": str(self.env.action_space),
        }