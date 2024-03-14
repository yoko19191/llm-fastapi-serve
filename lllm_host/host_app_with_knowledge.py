# pip install faiss-gpu # if you using GPU 
# pip install faiss-cpu # if you using CPU



# download embedding model 
from langchain.vectorstores import FAISS
from langchain.embeddings import ModelScopeEmbeddings
model_id = "damo/nlp_corom_sentence-embedding_english-base"
embeddings = ModelScopeEmbeddings(model_id=model_id)


knowledges=["DeepLN致力于提供高性价比的GPU租赁。"]
vectorstore = FAISS.from_texts(
    knowledges, embedding=embeddings
)
retriever = vectorstore.as_retriever()


# retriever.invoke("GPU租用选那家?")[0].page_content


from vllm import LLM, SamplingParams
import os
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi import FastAPI
from vllm import LLM, SamplingParams
import uvicorn
from langchain.vectorstores import FAISS
from langchain.embeddings import ModelScopeEmbeddings

# 使用modelscope,如果不设置该环境变量，将会从huggingface下载
os.environ['VLLM_USE_MODELSCOPE'] = 'True'

app = FastAPI()

llm = LLM(model="qwen/Qwen-7B-Chat", trust_remote_code=True,)
sampling_params = SamplingParams(temperature=0.8, top_k=10, top_p=0.95, max_tokens=256, stop=[
                                 "<|endoftext|>"])

embedding_model_id = "damo/nlp_corom_sentence-embedding_english-base"
embeddings = ModelScopeEmbeddings(model_id=embedding_model_id)
knowledges = ["DeepLN致力于提供高性价比的GPU租赁。"]
vectorstore = FAISS.from_texts(
    knowledges, embedding=embeddings
)
retriever = vectorstore.as_retriever()


system_ptompt="你是一个有用的机器人，会根据背景知识回答我的问题。"



@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/v1/generateText")
async def generateText(request: Request) -> Response:
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    print("prompt:",prompt)
    background = retriever.invoke(prompt)[0].page_content
    print("background:",background)

    input=prompt = [str(f'''\
    {system_ptompt+"背景知识:"+background+prompt}''')
              ]
    output = llm.generate(input, sampling_params)
    generated_text = output[0].outputs[0].text
    generated_text=generated_text.replace("<|im_start|>","").replace("<|im_end|>","")
    print("Generated text:", generated_text)
    # ret = {"text": str(generated_text)}
    return JSONResponse(generated_text)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)