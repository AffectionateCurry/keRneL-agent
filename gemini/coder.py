# File: /src/coder.py
import os
import openai
# Corrected import: extract_first_code is in kernelbench.src.utils
from KernelBench.src.utils import extract_first_code 
import logging

logger = logging.getLogger(__name__)

# It's good practice to use the new OpenAI client
try:
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except TypeError:  # Fallback for older openai versions if necessary, though new client is preferred
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    client = openai  # type: ignore
except openai.OpenAIError as e:  # Catch potential error if API key is missing during init
    logger.error(f"OpenAI API key not found or invalid. Please set OPENAI_API_KEY. Error: {e}")
    client = None


def gpt4o_code_generator(
    current_kernel_src: str,
    suggestion: str,
    blackbox_model_name: str = "gpt-4o",  # Or your preferred model
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> str:
    """
    Uses a blackbox LLM (e.g., GPT-4o) to apply a suggestion to the current kernel.

    Args:
        current_kernel_src: The source code of the current kernel (Kernel B).
        suggestion: The optimization suggestion from the Qwen model.
        blackbox_model_name: The name of the blackbox LLM to use.
        max_tokens: Max tokens for the generation.
        temperature: Temperature for generation.

    Returns:
        The source code of the new kernel (Kernel C), or original if generation fails.
    """
    if client is None:
        logger.error("OpenAI client not initialized. Cannot call GPT-4o.")
        return current_kernel_src  # Fallback

    prompt = f"""You are an expert CUDA code optimization assistant.
You are given a CUDA kernel and an optimization suggestion.
Your task is to apply the suggestion to the kernel and return the complete, updated Python module containing the new kernel.
Ensure the output is only the Python code block, with no other text or explanations.

Current Kernel:
```python
{current_kernel_src}
Use code with caution.
Python
Optimization Suggestion:
"{suggestion}"
Output only the updated Python code in a single code block:
"""
    try:
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):  # New API
            response = client.chat.completions.create(
                model=blackbox_model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for CUDA code optimization that only outputs Python code.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content if response.choices else ""
        else:  # Fallback for older openai library (less likely needed now)
            completion = openai.ChatCompletion.create(  # type: ignore
                model=blackbox_model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for CUDA code optimization that only outputs Python code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = completion.choices[0].message.content  # type: ignore

        if content:
            new_kernel_code = extract_first_code(content, ["python"])
            if new_kernel_code:
                logger.info(f"[GPT4oCoder] Successfully generated new kernel based on suggestion: {suggestion[:100]}...")
                return new_kernel_code

        logger.warning(f"[GPT4oCoder] Warning: Code generation failed or no code block found. Suggestion: {suggestion[:100]}...")
        return current_kernel_src  # Fallback to current kernel if generation fails
    except Exception as e:
        logger.error(f"[GPT4oCoder] Error during code generation: {e}", exc_info=True)
        return current_kernel_src  # Fallback