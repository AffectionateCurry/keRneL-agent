import os
import openai
from src.utils import extract_first_code # Assuming this function exists and works

# It's good practice to use the new OpenAI client
try:
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except TypeError: # Fallback for older openai versions if necessary, though new client is preferred
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    client = openai # type: ignore


def gpt4o_code_generator(
    current_kernel_src: str,
    suggestion: str,
    blackbox_model_name: str = "gpt-4o", # Or your preferred model
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
    prompt = f"""You are an expert CUDA code optimization assistant.
You are given a CUDA kernel and an optimization suggestion.
Your task is to apply the suggestion to the kernel and return the complete, updated Python module containing the new kernel.
Ensure the output is only the Python code block, with no other text or explanations.

Current Kernel:
```python
{current_kernel_src}

Optimization Suggestion:
"{suggestion}"

Output only the updated Python code in a single code block:
"""
    try:
        if hasattr(client, "chat") and hasattr(client.chat, "completions"): # New API
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
        else: # Fallback for older openai library (less likely needed now)
            completion = openai.ChatCompletion.create( # type: ignore
                model=blackbox_model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for CUDA code optimization that only outputs Python code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = completion.choices[0].message.content # type: ignore

        if content:
            new_kernel_code = extract_first_code(content, ["python"])
            if new_kernel_code:
                return new_kernel_code
        print(f"[GPT4oCoder] Warning: Code generation failed or no code block found. Suggestion: {suggestion[:100]}...")
        return current_kernel_src # Fallback to current kernel if generation fails
    except Exception as e:
        print(f"[GPT4oCoder] Error during code generation: {e}")
    return current_kernel_src # Fallback