import os
from typing import Optional
import openai
from KernelBench.src.utils import extract_first_code


class KernelCoder:
    """Manages code generation using blackbox LLMs."""
    
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def generate_kernel(
        self,
        current_kernel_src: str,
        suggestion: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """
        Apply an optimization suggestion to a kernel using GPT-4o.
        
        Args:
            current_kernel_src: Current kernel source code
            suggestion: Optimization suggestion from Qwen
            max_tokens: Maximum tokens for generation
            temperature: Generation temperature
            
        Returns:
            Updated kernel source code
        """
        prompt = self._create_prompt(current_kernel_src, suggestion)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert CUDA code optimization assistant. "
                                   "Output only complete Python code with CUDA kernels."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            content = response.choices[0].message.content
            if content:
                new_kernel = extract_first_code(content, ["python"])
                if new_kernel is None:
                    print("🚨 extract_first_code returned None; GPT content was:", content)
                if new_kernel:
                    return new_kernel
            
            print(f"[KernelCoder] Warning: Failed to extract code from response")
            return current_kernel_src
            
        except Exception as e:
            print(f"[KernelCoder] Error during generation: {e}")
            return current_kernel_src
    
    def _create_prompt(self, kernel_src: str, suggestion: str) -> str:
        """Create the prompt for kernel optimization."""
        return f"""Apply the following optimization suggestion to this CUDA kernel.
Return ONLY the complete updated Python module.

Current Kernel:
```python
{kernel_src}
Optimization Suggestion:
"{suggestion}"

**IMPORTANT**:  
– Wrap your new kernel inside a PyTorch `nn.Module` subclass named `ModelNew`.  
– Also define the helper functions `get_init_inputs()` and `get_inputs()` in the same module, mirroring the interface of the original `Model`.  
– Do not output anything else (no commentary, no text outside of the code block).

Updated Python module with optimized kernel:
```python"""