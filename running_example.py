import torch
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


llm = LLM(model="nvidia/NVIDIA-Nemotron-Nano-31B-A3-v3", trust_remote_code=True, enforce_eager=False)

outputs = llm.generate(prompts, moe_analyzer_save_dir = '/home/scratch.tbarnatan_gpu/repos/vLLM_SGLang_Llama3_Llama4_instructions/vllm/vllm-source/moe_analyzer.pt')
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")