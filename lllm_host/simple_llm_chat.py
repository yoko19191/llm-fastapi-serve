# pip install langchain vllm gptcache modelscope
# pip install transformers==4.32.0 accelerate tiktoken einops scipy transformers_stream_generator==0.0.4 peft deepspeed
# pip install auto-gptq optimum # if using int4 

from vllm import LLM, SamplingParams
import time
import os
#使用modelscope,如果不设置该环境变量，将会从huggingface下载
os.environ['VLLM_USE_MODELSCOPE']='True'

#无量化,最低显存占用约16.5GB
llm = LLM(model="qwen/Qwen-7B-Chat", trust_remote_code=True)
#int4量化,最低显存占用约7GB
# llm = LLM(model="qwen/Qwen-7B-Chat-int4", trust_remote_code=True,gpu_memory_utilization=0.35)

prompts = [
'''
Let's think step by step:
将大象塞到冰箱里面有几个步骤？
'''
]

sampling_params = SamplingParams(temperature=0.8,top_k=10, top_p=0.95,max_tokens=256,stop=["<|endoftext|>","<|im_end|>"])
start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
end_time = time.time()
latency = end_time - start_time
print(f"Latency: {latency} seconds")
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt} \nGenerated text: \n{generated_text}")

