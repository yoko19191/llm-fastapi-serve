# pip install -U fastapi
# pip install -U uvicorn  # an ASGI web server implementation for Python
# pip install -U openai-whisper
# pip install -U python-multipart

# from fastapi import FastAPI, UploadFile, HTTPException
# import whisper
# from tempfile import NamedTemporaryFile
# import os

# app = FastAPI()

# @app.post("/transcriptions")
# async def transcribe_audio(file: UploadFile, model: str = "tiny", language: str = None):
#     # validate model name 
#     if model not in ["tiny", "base", "small", "medium", "large", "large-v2"]:
#         raise HTTPException(status_code=400, detail=f"{model} not support.")
    
#     # tempfile of upload audio
#     try:
#         with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
#             contents = await file.read()
#             tmp_file.write(contents)
#             tmp_file_path = tmp_file.name
#         # load model and transcribe
#         try:
#             whisper_model = whisper.load_model(model)
#             result = whisper_model.transcribe(tmp_file_path, language=language)
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"transcribe fail: {str(e)}")
#         finally:
#             # delete temp file
#             if os.path.exists(tmp_file_path):
#                 os.remove(tmp_file_path)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"upload audio file fail: {str(e)}")

#     return {"text": result["text"]}


# -------------------------------
# or using faster-whisper as backend 
# pip install -U faster-whisper

from fastapi import FastAPI, UploadFile, HTTPException
from tempfile import NamedTemporaryFile
from faster_whisper import WhisperModel
import os 




# ---------

# Run on GPU with FP16
# """
# model = ['tiny']
# device = ['cpu', 'cuda']
# compute_type = ['float16', 'int8_float16', 'int8']
# """
model = "tiny"
model = WhisperModel(model, device="cpu", compute_type="int8") 


segments, info = model.transcribe("/Users/guchen/Downloads/audio_files/en_news.mp3", beam_size=10)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

print(info)

# print(type(segments))