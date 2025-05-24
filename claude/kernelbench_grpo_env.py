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
        qwen_system_prompt: str = """You are a senior CUDA performance engineer. Your job is to inspect the provided CUDA kernel code and suggest *TARGETED CUDA OPTIMIZATIONS*.

First, output your detailed chain of thought for arriving at the optimization. Explain *why* you are choosing a particular optimization.
Conclude your entire thought process with the exact line:
END_OF_THOUGHT

After the END_OF_THOUGHT line, and prefixed with "Actionable Suggestion:", provide ONLY the optimization suggestion in the specified pseudocode format.
The "Actionable Suggestion:" section must contain:
1. A single bullet describing the optimization name (e.g., loop fusion, shared-memory tiling, memory coalescing, unrolling).
2. A small fenced pseudocode snippet showing where/how to apply it (use “// pseudocode” comments).
3. A reference to the code region (e.g. “lines 32–45” or function name).

Do NOT emit actual CUDA or Python code—just high-level, structured steps of CUDA CODE.
Only output the bulleted list of pseudocode suggestions after "Actionable Suggestion:"—no extra commentary in that section.

Example Output Structure:
The kernel `kernel_b_src` exhibits poor data locality in the main loop. Accessing global memory repeatedly for `data[i]` can be a bottleneck. Introducing shared memory tiling for the loop processing `data` elements could improve performance by reducing global memory accesses and leveraging faster shared memory. I will focus on the loop at lines 10-15.
END_OF_THOUGHT
Actionable Suggestion:
- shared-memory tiling for the main processing loop
// pseudocode
// extern __shared__ float tile[];
// for (int i = blockIdx.x * blockSize_x * TILE_FACTOR + threadIdx.x; i < N; i += gridDim.x * blockSize_x * TILE_FACTOR) {
//   // Load a tile of data into shared memory `tile`
//   // Synchronize threads
//   // Compute using data from `tile`
//   // Synchronize threads before loading next tile (if necessary)
// }
lines 10-15
""",
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
            "max_new_tokens": 4096,
            "temperature": 1.0,
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

    def _parse_qwen_output(self, full_generated_text: str) -> (str, str):
        """
        Parses the Qwen output to separate chain of thought and actual suggestion.
        Returns: (chain_of_thought, actual_suggestion)
        """
        cot_content = ""
        actionable_suggestion = ""
        
        # Define the markers based on the system prompt
        thought_separator = "END_OF_THOUGHT"
        suggestion_prefix = "Actionable Suggestion:"

        separator_idx = full_generated_text.find(thought_separator)

        if separator_idx != -1:
            cot_content = full_generated_text[:separator_idx].strip()
            # Text after "END_OF_THOUGHT"
            remaining_text_after_cot = full_generated_text[separator_idx + len(thought_separator):].strip()
            
            suggestion_prefix_idx = remaining_text_after_cot.find(suggestion_prefix)
            if suggestion_prefix_idx != -1:
                # Extract text after "Actionable Suggestion:"
                actionable_suggestion = remaining_text_after_cot[suggestion_prefix_idx + len(suggestion_prefix):].strip()
            else:
                # If "END_OF_THOUGHT" is present but "Actionable Suggestion:" is not clearly found in the remainder,
                # we might assume the rest is the suggestion, or log a warning.
                actionable_suggestion = remaining_text_after_cot
                if actionable_suggestion:
                    print(f"Warning: '{suggestion_prefix}' not found after '{thought_separator}'. Using all subsequent text as suggestion: '{actionable_suggestion[:100]}...'")
                else:
                    print(f"Warning: No text found after '{thought_separator}' for suggestion. Model might have only output CoT.")
        else:
            # Fallback if "END_OF_THOUGHT" is not found.
            # Check if the output directly starts with "Actionable Suggestion:" (meaning no CoT or malformed CoT).
            direct_suggestion_prefix_idx = full_generated_text.find(suggestion_prefix)
            if direct_suggestion_prefix_idx == 0: # Output starts with the suggestion prefix
                cot_content = "" # No CoT detected
                actionable_suggestion = full_generated_text[len(suggestion_prefix):].strip()
                print(f"Warning: '{thought_separator}' not found. Output started directly with '{suggestion_prefix}'. Assuming no CoT.")
            elif direct_suggestion_prefix_idx > 0: # Suggestion prefix found, but not at start, and no separator
                cot_content = full_generated_text[:direct_suggestion_prefix_idx].strip() # Assume text before it is CoT
                actionable_suggestion = full_generated_text[direct_suggestion_prefix_idx + len(suggestion_prefix):].strip()
                print(f"Warning: '{thought_separator}' not found, but '{suggestion_prefix}' found later. Assuming text before '{suggestion_prefix}' is CoT.")
            else:
                # If neither marker is found as expected, treat the entire output as the suggestion.
                # This is a fallback to ensure we always have some actionable_suggestion.
                actionable_suggestion = full_generated_text
                cot_content = "" # Or you could set cot_content = full_generated_text if you suspect it's all CoT
                print(f"Warning: Neither '{thought_separator}' nor '{suggestion_prefix}' found as expected. Treating entire output as suggestion and CoT as empty/mixed: '{actionable_suggestion[:100]}...'")
        
        # Ensure actionable_suggestion is not empty if possible, otherwise use a placeholder.
        if not actionable_suggestion and full_generated_text:
            if cot_content == full_generated_text: # This means no suggestion part was identified.
                print(f"Warning: Parsed CoT is the full text. Actionable suggestion is empty. Using 'no_op'. Full text: {full_generated_text[:100]}")
                actionable_suggestion = "no_op" # Default action if suggestion is missing
            else: # CoT was separated, but suggestion part is empty.
                print(f"Warning: Actionable suggestion part is empty after parsing. Using 'no_op'. CoT: {cot_content[:100]}")
                actionable_suggestion = "no_op"

        elif not full_generated_text:
            print("Critical Warning: Model generated empty or whitespace-only text. Using 'no_op' suggestion.")
            actionable_suggestion = "no_op"
            cot_content = ""
            
        return cot_content, actionable_suggestion

    @modal.method()
    def generate_trajectory(
        self,
        agent_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ) -> Dict[str, List[Any]]:
        queries = []
        responses_for_grpo = []  # Stores the "actual output" (actionable suggestions)
        chains_of_thought = []   # Stores the CoT
        rewards = []

        obs, info = self.env.reset()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if self.generation_kwargs.get("pad_token_id") is None:
            self.generation_kwargs["pad_token_id"] = tokenizer.pad_token_id

        done = False
        while not done:
            prompt = self._create_prompt(obs)
            queries.append(prompt)

            messages = [
                {"role": "system", "content": self.qwen_system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            input_ids = None
            attention_mask = None

            try:
                if not hasattr(tokenizer, 'apply_chat_template'):
                    raise AttributeError("Tokenizer does not have apply_chat_template. Falling back to manual concatenation.")
                
                # Ensure model is on the correct device before tokenization if model.device is used by template
                # (though typically not an issue for apply_chat_template itself)
                model_device = agent_model.device if hasattr(agent_model, 'device') else 'cpu'

                tokenized_inputs = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_attention_mask=True
                )
                input_ids = tokenized_inputs.input_ids.to(model_device)
                if "attention_mask" in tokenized_inputs:
                    attention_mask = tokenized_inputs.attention_mask.to(model_device)
                else:
                    print("Warning: tokenizer.apply_chat_template did not explicitly return an attention_mask. Creating a default one.")
                    attention_mask = torch.ones_like(input_ids, device=model_device)
            
            except Exception as e_chat_template:
                print(f"Error using apply_chat_template: {e_chat_template}. Falling back to manual concatenation.")
                full_prompt_for_tokenizer = f"{self.qwen_system_prompt}\n\n{prompt}"
                model_device = agent_model.device if hasattr(agent_model, 'device') else 'cpu'
                
                tokenized_output = tokenizer.encode_plus(
                    full_prompt_for_tokenizer,
                    return_tensors="pt",
                    return_attention_mask=True,
                    truncation=True,
                    max_length=tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length else 4096 
                )
                input_ids = tokenized_output.input_ids.to(model_device)
                attention_mask = tokenized_output.attention_mask.to(model_device)

            if input_ids is None or attention_mask is None or input_ids.nelement() == 0:
                print("Error: Tokenization failed to produce valid input_ids or attention_mask. Skipping current step/trajectory.")
                # Handle error: skip step or end trajectory
                # For simplicity, let's assume we might want to end the trajectory if tokenization fails badly
                problem_name = info.get("problem_name", "unknown") if info else "unknown_problem_tokenization_error"
                final_speedup = info.get("speedup", 0.0) if info else 0.0
                # If this happens at the start, obs might not be updated, info could be from reset
                # If in a loop, info would be from last successful step.
                return { 
                    "queries": queries, "responses_for_grpo": responses_for_grpo, 
                    "chains_of_thought": chains_of_thought, "rewards": rewards, 
                    "problem_name": problem_name, 
                    "final_speedup": final_speedup,
                    "error": "Tokenization failure"
                }
            
            with torch.no_grad():
                # Ensure inputs are on the same device as the model
                input_ids = input_ids.to(agent_model.device)
                attention_mask = attention_mask.to(agent_model.device)
                output_ids = agent_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    **self.generation_kwargs
                )
            
            generated_ids_only = output_ids[0][input_ids.shape[1]:]
            full_generated_text = tokenizer.decode(generated_ids_only, skip_special_tokens=True).strip()
            
            # Log raw output for debugging if needed
            # print(f"--- Qwen Raw Output ---\n{full_generated_text}\n----------------------")

            cot_content, actionable_suggestion = self._parse_qwen_output(full_generated_text)
            
            # Log parsed parts for debugging
            # print(f"--- Parsed CoT ---\n{cot_content}\n--- Parsed Suggestion ---\n{actionable_suggestion}\n-------------------")

            chains_of_thought.append(cot_content)
            responses_for_grpo.append(actionable_suggestion)

            next_obs, reward, terminated, truncated, info = self.env.step(actionable_suggestion)
            rewards.append(reward)

            obs = next_obs
            done = terminated or truncated

        return {
            "queries": queries,
            "responses_for_grpo": responses_for_grpo,
            "chains_of_thought": chains_of_thought,
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
        trajectories = []
        for i in range(num_trajectories):
            print(f"Generating trajectory {i+1}/{num_trajectories}")
            try:
                trajectory = self.generate_trajectory(agent_model, tokenizer)
                trajectories.append(trajectory)
            except Exception as e:
                print(f"Error generating trajectory {i+1}: {e}")
                # Fallback to append an error placeholder for this trajectory
                trajectories.append({
                    "queries": [], "responses_for_grpo": [], "chains_of_thought": [], "rewards": [],
                    "problem_name": f"error_trajectory_{i+1}", "final_speedup": 0.0, "error_message": str(e)
                })
                continue
        return trajectories