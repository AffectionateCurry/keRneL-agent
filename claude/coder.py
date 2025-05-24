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
                        "content": """You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups.  YOU WILL  BE GIVEN SUGGESTIONS ON WHAT CUDA OPTIMIZATION YOU MAY NEED TO DO IN ORDER TO ACHIEVE BETTER PERFORMANCE (solving bank conflicts, shared memory, fusion, replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). ). Follow these suggestions to the best of your abilities. Your focus is to write CORRECT cuda code that correclty implements the suggested optimizations to achieve greater speedup. 



Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!



Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example given architecture is:

 

import torch

import torch.nn as nn

import torch.nn.functional as F





class Model(nn.Module):

    def __init__(self) -> None:

        super().__init__()



    def forward(self, a, b):

        return a + b





def get_inputs():

    # randomly generate input tensors based on the model architecture

    a = torch.randn(1, 128).cuda()

    b = torch.randn(1, 128).cuda()

    return [a, b]





def get_init_inputs():

    # randomly generate tensors required for initialization based on the model architecture

    return []





The example new arch with custom CUDA kernels looks like this: 



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.cpp_extension import load_inline



# Define the custom CUDA kernel for element-wise addition

elementwise_add_source = ""

#include <torch/extension.h>

#include <cuda_runtime.h>



__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {

        out[idx] = a[idx] + b[idx];

    }

}



torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {

    auto size = a.numel();

    auto out = torch::zeros_like(a);



    const int block_size = 256;

    const int num_blocks = (size + block_size - 1) / block_size;



    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);



    return out;

}

""



elementwise_add_cpp_source = (

    "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"

)



# Compile the inline CUDA code for element-wise addition

elementwise_add = load_inline(

    name="elementwise_add",

    cpp_sources=elementwise_add_cpp_source,

    cuda_sources=elementwise_add_source,

    functions=["elementwise_add_cuda"],

    verbose=True,

    extra_cflags=[""],

    extra_ldflags=[""],

)





class ModelNew(nn.Module):

    def __init__(self) -> None:

        super().__init__()

        self.elementwise_add = elementwise_add



    def forward(self, a, b):

        return self.elementwise_add.elementwise_add_cuda(a, b)





You are given the following architecture:



import torch

import torch.nn as nn



class Model(nn.Module):

    ""

    Model that performs a 3D transposed convolution, followed by a sum, layer normalization, average pooling, and GELU activation.

    ""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):

        super(Model, self).__init__()

        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)

        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))

        self.norm = nn.LayerNorm(norm_shape)

        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)

        self.gelu = nn.GELU()



    def forward(self, x):

        x = self.conv_transpose(x)

        x = x + self.sum_weight

        x = self.norm(x)

        x = self.avg_pool(x)

        x = self.gelu(x)

        return x



batch_size = 128

in_channels = 32

out_channels = 64

depth, height, width = 16, 32, 32

kernel_size = (3, 3, 3)

stride = (2, 2, 2)

padding = (1, 1, 1)

output_padding = (1, 1, 1)

sum_weight = 1.0

norm_shape = (out_channels,)

pool_kernel_size = (2, 2, 2)



def get_inputs():

    return [torch.randn(batch_size, in_channels, depth, height, width)]



def get_init_inputs():

    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size]

IMPORTANT: When you call load_inline, include + extra_cuda_cflags=["-gencode", "arch=compute_90,code=sm_90",], (and any other target archs) so the compiled kernel can actually run on NVIDIA H100.


"Output only complete Python code with CUDA kernels."


 """
                                   
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            content = response.choices[0].message.content
            print("🚀 GPT-4o raw output:\n" + content + "\n" + "-"*80)
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