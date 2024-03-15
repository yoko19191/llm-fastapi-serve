# LLM FastAPI Serve

在这个项目中你可以学习:

1. 如何预计一个大模型的推理和训练所需的显存大小
2. 如何使用Langchain 搭建一个LLM 的异步Host
3. 搭建一个 Whisper 的 异步Host 并实现负载均衡
4. 量化测试我们的RAG性能
5. 如果对我们的 大模型Host 进行压力测试
6. 探索如何使用 `accelerate` 和 `deepspeed` 进行分布式推理和部署优化
7. 使用 OneAPI 对外进行服务.

## 使用 `accelerate` 进行LLM 训练和推理显存大小估计

https://huggingface.co/docs/accelerate/main/en/usage_guides/model_size_estimator#caveats-with-this-calculator

https://techdiylife.github.io/blog/topic.html?category2=t05&blogid=0031

`accelerate` 是一个旨在简化用户大模型分布式训练和推理的python开发库, 搭配 HuggingFace 的其他生态比如 `transformers` 、`timm` 快速搭建测试LLM应用.

`accelerate` 提供了一个 CLI 接口方便用户快速估计模型的推理(实际预测的是加载模型所需显存大小)和训练的最低显存要求. 你可以使用 [model-memory calculator](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) 进行体验.

HuggingFace宣称, 在被 `transformers` 和 `timm` 库 支持的大模型, 其预测的误差大多在几个百分位之内.

我们将分别对已被 `transformers` 支持的大模型, 比如 [mistral-7b-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1),[ gemma-7b](https://huggingface.co/google/gemma-7b), [qwen-1.5-7b](https://huggingface.co/Qwen/Qwen1.5-7B), 和 未被支持的 [chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b) , [qwen-7b](https://huggingface.co/Qwen/Qwen-7B), [baichuan2-7b](https://huggingface.co/baichuan-inc/Baichuan-7B) 进行测试, 看看预测的实际误差.

安装 accelerate, transformers

```bash
pip install accelerate
pip install transformers
```

预测CLI:

```bash
accelerate estimate-memory mistralai/Mistral-7B-v0.1

accelerate estimate-memory mistralai/Mistral-7B-v0.1 --dtypes float16 # 只显示指定的数据类型

accelerate estimate-memory mistralai/Mistral-7B-v0.1 --dtypes float32 float16 --library_name transformers # 指定开发库

accelerate estimate-memory Qwen/Qwen-7B --trust_remote_code #设置 trust_remote_code=True
```

推理代码:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, pdb

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float32, device_map="auto")
#model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16, device_map="auto")
#model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", load_in_8bit=True, device_map="auto")
#model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", load_in_4bit=True, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
#pdb.set_trace()

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=1000)
print(tokenizer.decode(outputs[0]))
```

## 搭建一个异步 LLM Host

https://zhuanlan.zhihu.com/p/675834850?utm_campaign=&utm_medium=social&utm_psn=1751526554947932160

https://www.bilibili.com/video/BV1BC411z7nH/?spm_id_from=333.1296.top_right_bar_window_custom_collection.content.click&vd_source=427a8f6991c46f06262700ed0e9203dc

https://github.com/THUDM/ChatGLM3/tree/main/openai_api_demo


## 搭建一个异步 Whisper Host 并实现负载均衡

https://whisperapi.com/create-your-own-openai-whisper-speech-to-text-api

提供一个  transcriptions 端口允许音频文的 speech-to-text 


## 量化测试RAG性能

https://www.bilibili.com/video/BV1Jz421Q7Lw/?spm_id_from=333.1296.top_right_bar_window_custom_collection.content.click&vd_source=427a8f6991c46f06262700ed0e9203dc
