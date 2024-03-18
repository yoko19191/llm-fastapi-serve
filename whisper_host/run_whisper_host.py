# pip install -U fastapi
# pip install -U uvicorn 
# pip install -U openai-whisper
# pip install -U python-multipart
# https://huggingface.co/spaces/openai/whisper


"""
curl --request POST\
   --url 'http://127.0.0.1//transcriptions\
   -F "file=@YOUR_FILE_PATH"\
   -F "model=medium"\
"""


import uvicorn 

def run_localhost():
   uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)


if __name__ == "__main__":
   run_localhost()
