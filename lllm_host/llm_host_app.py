from vllm import LLM, SamplingParams
import os
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi import FastAPI
from vllm import LLM, SamplingParams
import uvicorn
#使用modelscope,如果不设置该环境变量，将会从huggingface下载
os.environ['VLLM_USE_MODELSCOPE']='True'

app = FastAPI()

llm = LLM(model="qwen/Qwen-7B-Chat", trust_remote_code=True)
sampling_params = SamplingParams(temperature=0.8,top_k=10, top_p=0.95,max_tokens=256,stop=["<|endoftext|>","<|im_end|>"])

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/v1/generateText")
async def generateText(request: Request) -> Response:
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    prompt=[f'''
    {prompt}
            '''
            ]
    print(prompt)
    output = llm.generate(prompt,sampling_params)
    generated_text=output[0].outputs[0].text
    print("Generated text:", generated_text)
    # ret = {"text": str(generated_text)}
    return JSONResponse(generated_text)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)