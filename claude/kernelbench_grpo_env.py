import torch
from typing import Dict, List, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
import modal
from claude.rl_kernel_env import KernelBenchRLEnv
GROUP_COL = "group_id"

class KernelBenchGRPOEnv:
    """GRPO-compatible wrapper for KernelBench RL environment with Modal support."""
    
    def __init__(
        self,
        rl_env_config: Dict[str, Any],
        trajectories_per_prompt: int = 16,           # 💡 new
        max_prompts_per_batch: int = 8,
        qwen_system_prompt: str = """You are a senior CUDA performance engineer. Your primary task is to analyze the **provided CUDA KERNEL SOURCE CODE** (the code within the `__global__` functions and related device functions, or the overall structure of a custom CUDA operator written in C++/CUDA for PyTorch) and suggest **TARGETED LOW-LEVEL CUDA KERNEL OPTIMIZATIONS**.

Your suggestions should focus on techniques that improve performance *directly within the CUDA kernel execution* or its interaction with GPU memory. Examples include:
- Shared memory tiling strategies
- Loop unrolling, fusion, or fission within the kernel
- Memory access pattern optimizations (coalescing, reducing bank conflicts)
- Instruction-level parallelism improvements
- Occupancy-improving launch configurations (if a kernel string is provided that includes a launch)
- Algorithm modifications *within the kernel* (e.g., parallel reduction patterns)
- Using CUDA intrinsics (e.g., warp-level primitives)
- Optimizing synchronization primitives (`__syncthreads()`)

**DO NOT suggest changes to the PyTorch host code (e.g., moving tensors to GPU with `.to('cuda')`, changing `torch.randn`, or modifying the `nn.Module` structure outside of the custom CUDA operator logic). Your focus is the CUDA C++ code of the kernel itself.**

Your output must have three parts in this specific order:

1.  **Thought Summary**: First, provide a very brief (1-2 sentence) summary of your main reasoning for choosing the specific CUDA kernel optimization. Prefix this summary with "Summary of My Reasoning:" and conclude it with the exact line:
    END_OF_REASONING_SUMMARY

2.  **Actionable Suggestion**: After END_OF_REASONING_SUMMARY, and prefixed with "Actionable Suggestion:", provide ONLY the *CUDA kernel optimization suggestion* in **pseudocode or descriptive text targeting the CUDA kernel code**.
    - If providing pseudocode, it should clearly show the intended change *within a CUDA kernel*.
    - Reference the specific functions within the *CUDA kernel source* that should be modified.

 **After providing the Actionable Suggestion, do not output any further text, code blocks, or newlines.**
""",
        qwen_prompt_template: str = """
## Context from Previous Step:
The kernel was in state 'A' (shown below if not the first step):
```cuda
{kernel_a_src}

Then, the following Actionable Suggestion was applied:
"{last_suggestion}"

The reasoning for that suggestion was summarized as:
"{summary_of_previous_reasoning}"

This resulted in the Current Kernel 'B' (THIS IS THE KERNEL TO OPTIMIZE NOW):
{kernel_b_src}

History of Suggestions and Outcomes in THIS Episode (Latest First):
{episode_attempts_history_formatted}
Example for history:
- Suggestion: "shared memory tiling for main loop" -> Achieved Speedup: 1.5x
- Suggestion: "loop unrolling of inner loop" -> Achieved Speedup: 1.2x

(If this is the first suggestion for the current kernel B, this section will state "No prior suggestions in this episode for the current kernel B.")

Your Task:
Based on the Current Kernel (B) and the historical context provided, generate:
Your detailed Chain of Thought (ending with END_OF_THOUGHT).
A brief Summary of My Reasoning (ending with END_OF_REASONING_SUMMARY).
An Actionable Suggestion (prefixed with Actionable Suggestion:).
Aim for a new and effective optimization for Current Kernel (B).

""",
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.env = KernelBenchRLEnv(**rl_env_config)
        self.qwen_system_prompt = qwen_system_prompt
        self.qwen_prompt_template = qwen_prompt_template
        self.trajectories_per_prompt = trajectories_per_prompt
        self.max_prompts_per_batch    = max_prompts_per_batch

        self.generation_kwargs = generation_kwargs or {
            "max_new_tokens": 4096,
            "temperature": 1.0,
            "top_p": 0.9,
            "do_sample": True,
        }

    def _create_prompt(
        self,
        observation: Dict[str, Any],
        last_cot_summary_for_current_step: str,
        episode_attempts_history: List[Dict[str, Any]]
    ) -> str:
        """Create Qwen prompt with observation, previous CoT summary, and episode history."""
        
        formatted_history = "No prior suggestions in this episode for the current kernel B."
        if episode_attempts_history:
            history_to_show = episode_attempts_history[-self.max_history_items_in_prompt:] # Get last N items
            history_items_formatted = []
            for attempt in reversed(history_to_show): # Show latest first
                suggestion_str = attempt.get('suggestion_text', 'N/A').split('\n')[0] # First line for brevity
                speedup_val = attempt.get('speedup', 'N/A')
                speedup_str = f"{speedup_val}"
                if isinstance(speedup_val, float):
                    speedup_str = f"{speedup_val:.2f}x"
                
                history_items_formatted.append(
                    f"- Suggestion: \"{suggestion_str}\" -> Achieved Speedup: {speedup_str}"
                )
            if history_items_formatted:
                formatted_history = "\n".join(history_items_formatted)

        return self.qwen_prompt_template.format(
            kernel_a_src=observation.get("kernel_a_src", "N/A (first step or error)"),
            kernel_b_src=observation.get("kernel_b_src", "N/A (error in environment state)"),
            last_suggestion=observation.get("last_suggestion", "N/A (first step)"),
            summary_of_previous_reasoning=last_cot_summary_for_current_step,
            episode_attempts_history_formatted=formatted_history,
        )

    def _parse_qwen_output(self, full_generated_text: str) -> (str, str, str):
        """
        Parses the Qwen output to separate chain of thought, CoT summary, and actual suggestion.
        Returns: (chain_of_thought, cot_summary, actual_suggestion)
        """
        cot_content = ""
        cot_summary = ""
        actionable_suggestion = ""

        thought_separator = "END_OF_THOUGHT"
        summary_prefix = "Summary of My Reasoning:" # Make sure this matches system prompt exactly
        summary_separator = "END_OF_REASONING_SUMMARY"
        suggestion_prefix = "Actionable Suggestion:" # Make sure this matches system prompt exactly

        thought_idx = full_generated_text.find(thought_separator)
        if thought_idx != -1:
            cot_content = full_generated_text[:thought_idx].strip()
            remaining_after_thought = full_generated_text[thought_idx + len(thought_separator):].strip()
            
            summary_start_idx = remaining_after_thought.find(summary_prefix)
            summary_end_idx = remaining_after_thought.find(summary_separator)

            if summary_start_idx != -1 and summary_end_idx != -1 and summary_start_idx < summary_end_idx:
                summary_text_start = summary_start_idx + len(summary_prefix)
                cot_summary = remaining_after_thought[summary_text_start:summary_end_idx].strip()
                remaining_after_summary = remaining_after_thought[summary_end_idx + len(summary_separator):].strip()
            elif summary_start_idx != -1: # Prefix found, but maybe not the end separator
                cot_summary = remaining_after_thought[summary_start_idx + len(summary_prefix):].strip()
                remaining_after_summary = "" # Assume rest is not suggestion if end separator missing
                print(f"Warning: Found '{summary_prefix}' but not '{summary_separator}'. Summary might be incomplete or malformed. Attempting to find suggestion next.")
                # Try to find suggestion in the original remaining_after_thought if summary parsing is iffy
                temp_suggestion_idx = remaining_after_thought.find(suggestion_prefix)
                if temp_suggestion_idx != -1 and temp_suggestion_idx > (summary_start_idx + len(summary_prefix) + len(cot_summary)): # ensure suggestion is after summary
                    remaining_after_summary = remaining_after_thought[temp_suggestion_idx:]


            else: # No summary prefix found
                remaining_after_summary = remaining_after_thought 
                print(f"Warning: '{summary_prefix}' not found after '{thought_separator}'. Assuming no CoT summary provided by model.")
            
            suggestion_idx = remaining_after_summary.find(suggestion_prefix)
            if suggestion_idx != -1:
                actionable_suggestion = remaining_after_summary[suggestion_idx + len(suggestion_prefix):].strip()
            else: # Suggestion prefix not found in the expected place
                actionable_suggestion = remaining_after_summary.strip() # Use whatever is left
                if actionable_suggestion:
                    print(f"Warning: '{suggestion_prefix}' not found after summary part. Using all remaining text as suggestion: '{actionable_suggestion[:100]}...'")
                else:
                    print(f"Warning: No text found after summary part for suggestion.")
        else: # No END_OF_THOUGHT found
            print(f"Warning: '{thought_separator}' not found. Attempting to parse for suggestion directly.")
            direct_suggestion_idx = full_generated_text.find(suggestion_prefix)
            if direct_suggestion_idx != -1:
                actionable_suggestion = full_generated_text[direct_suggestion_idx + len(suggestion_prefix):].strip()
                cot_content = full_generated_text[:direct_suggestion_idx].strip() # Treat anything before as CoT
                print(f"Info: Assuming text before direct suggestion is CoT: '{cot_content[:100]}...'")
            else:
                actionable_suggestion = full_generated_text # Ultimate fallback
                print(f"Warning: No structure (CoT, Summary, Suggestion) found. Treating entire output as suggestion: '{actionable_suggestion[:100]}...'")
        
        if not actionable_suggestion and full_generated_text:
            print(f"Warning: Actionable suggestion is empty after parsing attempts. Using 'no_op'. Full text: '{full_generated_text[:100]}...'")
            actionable_suggestion = "no_op"
        elif not full_generated_text:
            print("Critical Warning: Model generated empty or whitespace-only text. Using 'no_op'.")
            actionable_suggestion = "no_op"
            cot_content = ""
            cot_summary = ""
            
        return cot_content, cot_summary, actionable_suggestion

    @modal.method()
    def generate_trajectory(
        self,
        agent_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ) -> Dict[str, List[Any]]:
        queries = []
        responses_for_grpo = []
        cot_summaries_log = [] 
        rewards_log = []
        full_interaction_log = []

        obs, initial_info = self.env.reset()
        
        last_cot_summary = "N/A (first step in episode)"
        episode_attempts_history_list = []

        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        if self.generation_kwargs.get("pad_token_id") is None:
            self.generation_kwargs["pad_token_id"] = tokenizer.pad_token_id

        done = False
        current_step_in_episode = 0
        info_for_return = initial_info # Keep track of the latest info

        while not done:
            current_step_in_episode += 1
            prompt = self._create_prompt(
                observation=obs,
                last_cot_summary_for_current_step=last_cot_summary,
                episode_attempts_history=episode_attempts_history_list
            )
            queries.append(prompt)

            messages = [
                {"role": "system", "content": self.qwen_system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            input_ids, attention_mask = None, None
            try:
                # ... (your existing tokenization logic, make sure it uses agent_model.device) ...
                model_device = agent_model.device if hasattr(agent_model, 'device') else 'cpu'
                if not hasattr(tokenizer, 'apply_chat_template'):
                    raise AttributeError("Tokenizer does not have apply_chat_template.")
                
                tokenized_inputs = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt", return_attention_mask=True
                )
                input_ids = tokenized_inputs.input_ids.to(model_device)
                attention_mask = tokenized_inputs.attention_mask.to(model_device) if "attention_mask" in tokenized_inputs else torch.ones_like(input_ids, device=model_device)
            
            except Exception as e_chat_template:
                print(f"Error using apply_chat_template: {e_chat_template}. Falling back to manual concat.")
                full_prompt_for_tokenizer = f"{self.qwen_system_prompt}\n\n{prompt}"
                model_device = agent_model.device if hasattr(agent_model, 'device') else 'cpu'
                tokenized_output = tokenizer.encode_plus(
                    full_prompt_for_tokenizer, return_tensors="pt", return_attention_mask=True,
                    truncation=True, max_length=tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length else 4096
                )
                input_ids = tokenized_output.input_ids.to(model_device)
                attention_mask = tokenized_output.attention_mask.to(model_device)


            if input_ids is None or attention_mask is None or input_ids.nelement() == 0:
                print("Error: Tokenization failed. Ending trajectory.")
                # Populate with placeholders if needed for consistent GRPO data format
                if not responses_for_grpo: responses_for_grpo.append("tokenization_failure")
                if not rewards_log: rewards_log.append(info_for_return.get("reward", -1.0))
                if not cot_summaries_log: cot_summaries_log.append("N/A")
                break 

            with torch.no_grad():
                output_ids = agent_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    **self.generation_kwargs
                )
            
            generated_ids_only = output_ids[0][input_ids.shape[1]:]
            full_generated_text = tokenizer.decode(generated_ids_only, skip_special_tokens=True).strip()
            
            _full_cot, current_cot_summary, actionable_suggestion = self._parse_qwen_output(full_generated_text)
            
            if not actionable_suggestion or actionable_suggestion == "no_op":
                print(f"Warning: Empty or no_op suggestion from Qwen at step {current_step_in_episode}. Ending episode.")
                if not responses_for_grpo : responses_for_grpo.append("qwen_no_op_suggestion")
                if not rewards_log : rewards_log.append(info_for_return.get("reward", -1.0)) # Penalize
                if not cot_summaries_log : cot_summaries_log.append(current_cot_summary or "N/A_no_op")
                break

            responses_for_grpo.append(actionable_suggestion)
            cot_summaries_log.append(current_cot_summary if current_cot_summary else "N/A_no_summary_parsed")

            next_obs, reward, terminated, truncated, info_after_step = self.env.step(actionable_suggestion)
            rewards_log.append(reward)
            info_for_return = info_after_step # Update with the latest info

            full_interaction_log.append({
                "step": current_step_in_episode, "prompt_to_qwen": prompt,
                "full_qwen_output": full_generated_text, "parsed_cot_summary": current_cot_summary,
                "parsed_actionable_suggestion": actionable_suggestion,
                "kernel_a_src_in_prompt": obs.get("kernel_a_src"),
                "kernel_b_src_in_prompt": obs.get("kernel_b_src"),
                "reward_from_env": reward, "speedup_from_env": info_after_step.get("speedup", 0.0),
                "kernel_b_after_suggestion": next_obs.get("kernel_b_src")
            })

            episode_attempts_history_list.append({
                "suggestion_text": actionable_suggestion,
                "speedup": info_after_step.get("speedup", "N/A"),
                "reward": reward
            })
            
            last_cot_summary = current_cot_summary if current_cot_summary else "N/A (previous step had no summary)"
            obs = next_obs
            done = terminated or truncated
            
            if done:
                print(f"Episode finished after {current_step_in_episode} steps. Terminated: {terminated}, Truncated: {truncated}")


        # Ensure all lists have the same length as queries for GRPO trainer,
        # especially if loop broke early. GRPOTrainer expects queries and responses to align.
        # However, responses_for_grpo and rewards_log should naturally align with actual steps taken.
        # The `queries` list will have one extra item if the loop broke due to Qwen's bad output
        # before env.step() for the last query.
        # For GRPO, the important alignment is between the query that LED to a response, and that response and its reward.

        num_actual_steps = len(responses_for_grpo)
        final_queries = queries[:num_actual_steps] # Only queries that got a response and a reward

        # If an error occurred and lists are empty, provide some default to avoid crashing trainer
        if not final_queries and queries: # E.g. first step tokenization failed badly
            final_queries = [queries[0]] # Or a placeholder query
            if not responses_for_grpo: responses_for_grpo = ["initialization_error_response"]
            if not rewards_log: rewards_log = [-1.0] # Default penalty
            if not cot_summaries_log: cot_summaries_log = ["N/A"]


        return {
            "queries": final_queries, # Use queries that led to actionable steps
            "responses_for_grpo": responses_for_grpo,
            # "cot_summaries": cot_summaries_log, # Full log of summaries
            "rewards": rewards_log,
            "problem_name": info_for_return.get("problem_name", "unknown"),
            "final_speedup": info_for_return.get("speedup", 0.0),
            # "full_interaction_log_per_step": full_interaction_log # Optional for debugging
        }

    @modal.method()
    def batch_generate_trajectories(
        self, agent_model, tokenizer
    ) -> List[Dict]:
        """Produce `max_prompts_per_batch` problems,
        each with `trajectories_per_prompt` independent trajectories."""

        batch: List[Dict[str, Any]] = []

        for prompt_idx in range(self.max_prompts_per_batch):
            # -------- Fix problem choice so every inner loop sees same prompt
            first_obs, info0 = self.env.reset(problem_idx=self.env.current_problem_idx)

            for k in range(self.trajectories_per_prompt):
                print(f"Problem {prompt_idx+1}/{self.max_prompts_per_batch} – "
                      f"trajectory {k+1}/{self.trajectories_per_prompt}")

                traj = self.generate_trajectory(agent_model, tokenizer)

                # Tag every completion in this trajectory with the SAME group id
                gid = prompt_idx                # 0 … 7
                traj[GROUP_COL] = [gid] * len(traj["rewards"])

                batch.append(traj)

        return batch