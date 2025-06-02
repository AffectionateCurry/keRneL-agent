# claude/kernelbench_grpo_env.py
# --- BEGIN MODIFICATIONS ---
import torch
from typing import Dict, List, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
import modal
import traceback
import hashlib # For creating history item IDs
import re

GROUP_COL = "group_id"

from .rl_kernel_env import KernelBenchRLEnv # Assumes it's in the same dir
from KernelBench.src.utils import extract_first_code # Ensure this is importable

GROUP_COL = "group_id"

class KernelBenchGRPOEnv:
    GROUP_COL = GROUP_COL 
    """
    GRPO-compatible wrapper for KernelBench RL environment.
    Qwen is prompted to generate full Python/CUDA code modules.
    """
    
    def __init__(
        self,
        rl_env_config: Dict[str, Any],
        trajectories_per_prompt: int = 16,
        max_prompts_per_batch: int = 8,
        # Updated Qwen System Prompt:
        qwen_system_prompt: str = """


You are an expert senior CUDA programmer.
Your task is to rewrite a given PyTorch module (“Kernel B”) as a *fully
self-contained* Python file named **ModelNew** that offloads heavy work to
custom CUDA kernels via `torch.utils.cpp_extension.load_inline`.

──────────────────────────────────────────────────
OUTPUT FORMAT  –  STRICT CONTRACT
──────────────────────────────────────────────────

2. The evaluator only uses what sits between
      <final_answer>  …  </final_answer>

3. Inside that tag pair you must include **one** fenced block

      ```python
      # complete ModelNew module
      ```

4. Nothing-at-all may appear outside those tags.

Copy this schema exactly in every answer:

<final_answer>
```python
# …complete ModelNew code here…
</final_answer>

──────────────────────────────────────────────────
TECHNICAL REQUIREMENTS
──────────────────────────────────────────────────
■ Keep identical semantics to the original model.
■ Define CUDA kernels and their C++ wrappers.
■ Always pass the second positional arg of load_inline
(cpp_sources), even if it is just "".
■ Add extra_cuda_cflags=["-gencode","arch=compute_90,code=sm_90"]
(and any others you need) so the binary runs on H100.
■ Provide get_inputs() and get_init_inputs() that match the new model.

──────────────────────────────────────────────────
WORKED EXAMPLE — FOLLOW THE PATTERN
──────────────────────────────────────────────────
Everything below would live inside your <final_answer> block.

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_kernel_with_wrapper = r""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a,
                                       const float* b,
                                       float* out,
                                       int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    int n = a.numel();
    auto out = torch::empty_like(a);
    int blk = 256, grd = (n + blk - 1) / blk;
    elementwise_add_kernel<<<grd, blk>>>(a.data_ptr<float>(),
                                         b.data_ptr<float>(),
                                         out.data_ptr<float>(),
                                         n);
    return out;
}
""

cpp_proto = "torch::Tensor elementwise_add_cuda(torch::Tensor,torch::Tensor);"

elementwise_add = load_inline(
    name        = "elementwise_add",
    cpp_sources = cpp_proto,              # never omit this
    cuda_sources= cuda_kernel_with_wrapper,
    functions   = ["elementwise_add_cuda"],
    verbose     = False,
    extra_cuda_cflags=["-gencode","arch=compute_90,code=sm_90"],
)

class ModelNew(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return elementwise_add.elementwise_add_cuda(a, b)

def get_inputs():
    return [torch.randn(1, 128, device="cuda"),
            torch.randn(1, 128, device="cuda")]

def get_init_inputs():
    return []
──────────────────────────────────────────────────
Now optimise the Kernel B you receive in the user prompt.
Remember: your reply must contain exactly ONE <final_answer> block, as
shown above, and nothing else.


"""      ,
        # Updated Qwen User Prompt Template:
        qwen_prompt_template: str = """
        Current PyTorch Module to Optimize ('Kernel B'):
        {kernel_b_src}
        Context from Previous State ('Kernel A'):
        (This is the state of the module before it became 'Kernel B'. If this is the first step, Kernel A and Kernel B are identical to the original problem.)
        {kernel_a_src}
        Information about the last code generation attempt:
        {last_generated_code_info}
        History of Optimization Attempts in THIS Episode (Latest First):
        {episode_attempts_history_formatted}
        Your Task:
Based on 'Kernel B' and the historical context, generate the complete CUDA  code for the optimized 'ModelNew' module, following all instructions from the system prompt.
– Also define the helper functions `get_init_inputs()` and `get_inputs()` in the same module, mirroring the interface of the original `Model`.  ARE YOU STUPID. PLEASE ONLY GENERATE EXACTLY *ONE* KERNEL THAT YOU THINK WILL PERFORM THE BEST. DO NOT OUTPUT ANYTHING ELSE BESIDES THIS KERNEL (NO COMMENTARY OR TEXT OUTSIDE THE KERNEL)
REMEMBER: Your reply **must exactly follow** the schema below  
 (everything, including the opening and closing tags, must be in *your* reply):  

 <final_answer>
 ```python
 # …your complete ModelNew code here…
 ```
 </final_answer>

""",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        max_history_items_in_prompt: int = 3, # Max history items to show in prompt
        ):
        # rl_env_config should NOT contain 'kernel_coder' anymor
        self.env = KernelBenchRLEnv(**rl_env_config)
        self.qwen_system_prompt = qwen_system_prompt
        self.qwen_prompt_template = qwen_prompt_template
        self.trajectories_per_prompt = trajectories_per_prompt
        self.max_prompts_per_batch    = max_prompts_per_batch
        self.max_history_items_in_prompt = max_history_items_in_prompt


        self.generation_kwargs = generation_kwargs or {
            "max_new_tokens": 8192,  # Increased significantly for full code
            "temperature": 0.7,      # Adjusted for code generation
            "top_p": 0.95,
            "do_sample": True,
            "repetition_penalty": 1.05,
    }

    def _create_prompt(
        self,
        observation: Dict[str, Any], # From env.reset() or env.step()
        episode_attempts_history: List[Dict[str, Any]]
    ) -> str:
        formatted_history = "No prior optimization attempts in this episode."
        if episode_attempts_history:
            history_to_show = episode_attempts_history[-self.max_history_items_in_prompt:]
            history_items_formatted = []
            for attempt in reversed(history_to_show): # Show latest first
                code_hash = attempt.get('code_hash_short', 'N/A')
                compile_status = attempt.get('compiled', 'N/A')
                correct_status = attempt.get('correctness', 'N/A')
                speedup_val = attempt.get('speedup', 'N/A')
                reward_val = attempt.get('reward', 'N/A')

                speedup_str = f"{speedup_val:.2f}x" if isinstance(speedup_val, float) else str(speedup_val)
                reward_str = f"{reward_val:.3f}" if isinstance(reward_val, float) else str(reward_val)
                
                history_items_formatted.append(
                    f"- Code (Hash: {code_hash}) -> Result: Compiled={compile_status}, Correct={correct_status}, Speedup: {speedup_str}, Reward: {reward_str}"
                )
            if history_items_formatted:
                formatted_history = "\n".join(history_items_formatted)

        return self.qwen_prompt_template.format(
            kernel_a_src=observation.get("kernel_a_src", "N/A (initial state)"),
            kernel_b_src=observation.get("kernel_b_src", "N/A (initial state, error?)"),
            last_generated_code_info=observation.get("last_generated_code_info", "N/A (first step)"),
            episode_attempts_history_formatted=formatted_history,
        )

    def _parse_qwen_output(self, full_generated_text: str, original_kernel_b_src: str) -> str:
        if not full_generated_text or full_generated_text.isspace():
            # ...
            return original_kernel_b_src

        match = re.search(r"<\s*final_answer\s*>(.*?)</\s*final_answer\s*>", full_generated_text, re.DOTALL | re.IGNORECASE)
        extracted_code = None # Initialize

        if match:
            final_answer_content = match.group(1).strip()
            print("Successfully extracted content from <final_answer> tags.")
            
            # OPTION B: Assume final_answer_content IS the Python code, if LLM doesn't use fences there.
            # Add a basic check to see if it looks like code.
            if final_answer_content.startswith("import ") or \
            final_answer_content.startswith("#") or \
            final_answer_content.startswith("class ModelNew"): # Be more specific if needed
                print("DEBUG: Using content within <final_answer> directly as Python code.")
                extracted_code = final_answer_content
            else:
                # This else block means it didn't look like raw code, 
                # OR you want to always try fenced extraction first.
                print("DEBUG: Content in <final_answer> didn't immediately look like raw code, or trying fenced extraction first.")
                print(f"--- DEBUG: Content being passed to extract_first_code from <final_answer> ---")
                print(repr(final_answer_content))
                print(f"--- END DEBUG ---")
                extracted_code = extract_first_code(final_answer_content, ["python"])
                if extracted_code:
                    print("DEBUG: Successfully extracted code with fences from <final_answer> content.")
                else:
                    print("DEBUG: Failed to extract code with fences from <final_answer> content.")


        else: # <final_answer> tags not found
            print(f"Warning: Could not find <final_answer> tags. Falling back to extracting first code block from entire output.")
            extracted_code = extract_first_code(full_generated_text, ["python"])
            if not extracted_code:
                print("DEBUG: Fallback extraction from full text also failed.")

        # Now, validate whatever 'extracted_code' we ended up with (or if it's still None)
        if extracted_code:
            if "class ModelNew" in extracted_code :
                print("Successfully validated 'ModelNew' code.")
                return extracted_code
            else:
                print(f"Warning: Code (from <final_answer> or fallback) doesn't seem to be a complete 'ModelNew' module. Using original_kernel_b_src.")
                print(f"--- BEGIN INVALID CODE ---")
                print(extracted_code)
                print(f"--- END INVALID CODE ---")
                return original_kernel_b_src
        else:
            print(f"Warning: No Python code could be extracted. Using original_kernel_b_src.")
            print(f"--- BEGIN FULL QWEN OUTPUT (NO CODE EXTRACTED) ---\n{full_generated_text}\n--- END ---")
            return original_kernel_b_src
    @modal.method()
    def generate_trajectory(
        self,
        agent_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        problem_idx_for_env_reset: Optional[int] = None, # <<< ADDED this argument
    ) -> Dict[str, List[Any]]:
        queries = []
        responses_for_grpo = []
        cot_summaries_log = []
        rewards_log = []
        full_interaction_log = []

        # Pass problem_idx_for_env_reset to self.env.reset()
        obs, initial_info = self.env.reset(problem_idx=problem_idx_for_env_reset) # <<< MODIFIED THIS LINE
        
        # last_cot_summary was used for _create_prompt before, but _create_prompt doesn't take it.
        # It seems it was intended to be related to the history or last step's info.
        # For now, its direct use in prompt creation is removed as per Solution 1.
        # If it's needed for other logic, it can be kept.
        # For instance, `obs['last_generated_code_info']` now carries info from the previous step.

        episode_attempts_history_list = [] # Renamed from episode_attempts_history for clarity

        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        if self.generation_kwargs.get("pad_token_id") is None:
            self.generation_kwargs["pad_token_id"] = tokenizer.pad_token_id

        done = False
        current_step_in_episode = 0
        info_for_return = initial_info

        while not done:
            current_step_in_episode += 1
            
            # Corrected call to _create_prompt (as per Solution 1 from previous discussion)
            prompt = self._create_prompt(
                observation=obs,
                episode_attempts_history=episode_attempts_history_list
            )
            queries.append(prompt)

            messages = [
                {"role": "system", "content": self.qwen_system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            input_ids, attention_mask = None, None
            try:
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
                effective_max_length = tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length else 4096
                tokenized_output = tokenizer.encode_plus(
                    full_prompt_for_tokenizer, return_tensors="pt", return_attention_mask=True,
                    truncation=True, max_length=effective_max_length
                )
                input_ids = tokenized_output.input_ids.to(model_device)
                attention_mask = tokenized_output.attention_mask.to(model_device)

            if input_ids is None or attention_mask is None or input_ids.nelement() == 0:
                print("Error: Tokenization failed. Ending trajectory.")
                if not responses_for_grpo: responses_for_grpo.append("tokenization_failure")
                if not rewards_log: rewards_log.append(info_for_return.get("reward", -1.0)) # Use last known reward or penalty
                if not cot_summaries_log: cot_summaries_log.append("N/A_tokenization_failure")
                break 

            with torch.no_grad():
                output_ids = agent_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    **self.generation_kwargs
                )
            
            generated_ids_only = output_ids[0][input_ids.shape[1]:]
            full_generated_text = tokenizer.decode(generated_ids_only, skip_special_tokens=True).strip()

            print("\n===== RAW LLM OUTPUT BEGIN =====\n")
            print(full_generated_text)
            print("\n===== RAW LLM OUTPUT END =====\n")


            
            # --- BEGIN: APPLICATION OF SOLUTION 2 ---
            current_kernel_b_src_for_parsing = obs.get("kernel_b_src")
            if current_kernel_b_src_for_parsing is None:
                print(f"CRITICAL ERROR: obs['kernel_b_src'] is None before calling _parse_qwen_output. "
                      f"This can happen if env.reset() or env.step() failed to populate obs correctly. "
                      f"Using an empty string as fallback, but this is a serious issue. Obs: {obs}")
                current_kernel_b_src_for_parsing = "" 

            # Correct call to _parse_qwen_output, expecting one return value.
            # Pass the original kernel_b_src that the LLM was trying to optimize.
            actionable_suggestion = self._parse_qwen_output(
                full_generated_text,
                original_kernel_b_src=current_kernel_b_src_for_parsing
            )

            # Since the prompt asks for "Output ONLY the complete Python code",
            # there's no separate CoT or summary to extract from the LLM's main output.
            current_cot_summary = "N/A (Code-only generation)" # Placeholder for logging

            # Check if parsing failed (returned original), or suggestion is empty,
            # or if the suggestion is literally the same as what was input (no effective change).
            # The _parse_qwen_output returns original_kernel_b_src on parse failure.
            if not actionable_suggestion or actionable_suggestion == current_kernel_b_src_for_parsing:
                print(f"Warning: Qwen output parsing failed, was empty, or resulted in no change from original kernel_b_src "
                      f"at step {current_step_in_episode}. Treating as failed suggestion. "
                    )
                
                
                # A query was made, but the response was unusable.
                # We need to append placeholder/penalty values for this step.
                # responses_for_grpo, rewards_log, cot_summaries_log should have same length as actual attempts.
                
                # No valid response was generated for the last query.
                # For GRPO, we need a response and reward for every query that participated in a step.
                # If we break here, the last query in 'queries' won't have a corresponding response/reward
                # from an env.step(). We should still log a "response" and "reward" for this failed generation.

                # Add a placeholder response indicating failure
                responses_for_grpo.append("qwen_parse_failure_or_no_change")
                # Assign a penalty reward for this failed generation attempt
                rewards_log.append(-1.0) # Default penalty
                # Log the placeholder summary
                cot_summaries_log.append(current_cot_summary)
                break # End the episode because the LLM output was not usable
            # --- END: APPLICATION OF SOLUTION 2 ---

            responses_for_grpo.append(actionable_suggestion)
            cot_summaries_log.append(current_cot_summary) # Log the placeholder summary for successful parse

            next_obs, reward, terminated, truncated, info_after_step = self.env.step(actionable_suggestion)
            rewards_log.append(reward) # This is the reward for the step taken with actionable_suggestion
            info_for_return = info_after_step

            # Update episode history using info from AFTER the step
            # Generate a short hash for the suggested code for history display
            code_hash_short = hashlib.md5(actionable_suggestion.encode()).hexdigest()[:6]
            episode_attempts_history_list.append({
                "code_hash_short": code_hash_short, # For brevity in prompt
                # "suggestion_text": actionable_suggestion, # Too long for prompt history
                "compiled": info_after_step.get("eval_result", {}).get("compiled", "N/A"),
                "correctness": info_after_step.get("eval_result", {}).get("correctness", "N/A"),
                "speedup": info_after_step.get("speedup", "N/A"),
                "reward": reward
            })
            
            # last_cot_summary for the next step's prompt creation is not directly used now.
            # The observation `obs` (which becomes `next_obs`) contains `last_generated_code_info`
            # which serves a similar purpose.
            obs = next_obs
            done = terminated or truncated
            
            if done:
                print(f"Episode finished after {current_step_in_episode} steps. Terminated: {terminated}, Truncated: {truncated}")

        # Ensure all lists have the same length as queries for GRPO trainer.
        # The number of actual steps taken where a response was processed by the environment
        # is len(responses_for_grpo), which should also be len(rewards_log).
        num_actual_steps = len(responses_for_grpo)
        final_queries = queries[:num_actual_steps]

        if not final_queries and queries: # E.g. first step tokenization failed, or first LLM call failed before step
            if not responses_for_grpo: responses_for_grpo = ["initialization_or_first_step_error_response"]
            if not rewards_log: rewards_log = [-1.0] 
            if not cot_summaries_log: cot_summaries_log = ["N/A_initial_error"]
            # If queries has one item due to prompt creation, but then tokenization/generation failed immediately
            if len(queries) == 1 and num_actual_steps == 0:
                 # This means the loop didn't even run once for a step.
                 # The `break` conditions inside the loop already handle appending to responses/rewards/cot_summaries.
                 # This 'if' block might be more for cases where `input_ids is None` break happens.
                 # Let's ensure final_queries matches the length of responses_for_grpo.
                 if len(responses_for_grpo) == 1: # If the break condition inside the loop added one item
                     final_queries = [queries[0]]
                 else: # Should not happen if break conditions are correct
                     final_queries = [] # Or handle as error

        # Defensive check: all lists for GRPO should have same length
        min_len = min(len(final_queries), len(responses_for_grpo), len(rewards_log))
        final_queries = final_queries[:min_len]
        responses_for_grpo = responses_for_grpo[:min_len]
        rewards_log = rewards_log[:min_len]
        # cot_summaries_log is for internal logging, not directly for GRPOTrainer usually

        return {
            "queries": final_queries,
            "responses_for_grpo": responses_for_grpo,
            "rewards": rewards_log,
            "problem_name": info_for_return.get("problem_name", "unknown"), # Use problem_name from env
            "final_speedup": info_for_return.get("speedup", 0.0),
            # "full_interaction_log_per_step": full_interaction_log # Optional for debugging
            # "cot_summaries": cot_summaries_log, # If needed for other analysis
        }
    @modal.method()
    def batch_generate_trajectories(
        self, agent_model, tokenizer
    ) -> List[Dict[str, Any]]: # Return type is List of trajectory dicts
        """
        Generates a batch of trajectories.
        Each "prompt" (KernelBench problem) gets `trajectories_per_prompt` attempts.
        """
        batch_trajectories: List[Dict[str, Any]] = []
        if not self.env.problem_indices:
            print("Error: No problem indices available in the environment.")
            return []
            
        available_problem_indices_in_env = self.env.problem_indices

        for i in range(self.max_prompts_per_batch):
            # Determine the actual problem index from KernelBench dataset for this group
            # self.env.current_problem_idx is an index into available_problem_indices_in_env
            if self.env.current_problem_idx >= len(available_problem_indices_in_env):
                print(f"Warning: current_problem_idx ({self.env.current_problem_idx}) is out of bounds for available_problem_indices_in_env. Resetting.")
                self.env.current_problem_idx = 0
            
            # This is the index to pass to env.reset(), which expects an index from the full dataset
            actual_dataset_problem_idx = available_problem_indices_in_env[self.env.current_problem_idx] 
            
            for k in range(self.trajectories_per_prompt):
                print(f"Problem Group {i+1}/{self.max_prompts_per_batch} (Dataset Problem Idx: {actual_dataset_problem_idx}) – Trajectory {k+1}/{self.trajectories_per_prompt}")

                traj_data = self.generate_trajectory(
                    agent_model, 
                    tokenizer,
                    problem_idx_for_env_reset=actual_dataset_problem_idx # Pass it here
                )

                # Tag every (query,response,reward) tuple in this trajectory with the SAME group_id
                # This is crucial for GRPO
                if traj_data["queries"]: # Only add if trajectory is not empty
                    gid = i # Group ID is based on the problem_prompt index
                    traj_data[GROUP_COL] = [gid] * len(traj_data["rewards"])
                    batch_trajectories.append(traj_data)
                else:
                    print(f"Warning: Empty trajectory generated for problem group {i+1}, traj {k+1}. Skipping.")
            
            # Advance the environment's problem pointer for the NEXT group of trajectories
            self.env.current_problem_idx = (self.env.current_problem_idx + 1) % len(available_problem_indices_in_env)
        
        return batch_trajectories