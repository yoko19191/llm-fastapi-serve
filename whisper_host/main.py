# pip install -U fastapi
# pip install -U python-multipart
# pip install -U uvicorn  # an ASGI web server implementation for Python
# pip install -U openai-whisper


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
from fastapi.responses import StreamingResponse
from tempfile import NamedTemporaryFile
from faster_whisper import WhisperModel
import os 


"""
model = ['base', 'base.en', 'tiny', 'tiny.en', 'small', 'small.en', 'distil-small.en', \\
      'medium', 'medium.en', 'distil-medium.en', 'large-v1', 'large-v2', 'distil-large-v2', 'large-v3']
device = ['cpu', 'cuda']
compute_type = ['float16', 'int8_float16', 'int8']
"""

# model = "large-v3"
# model = WhisperModel(model, device="cuda", compute_type="int8") 

# upload_audio_path = r"C:\Users\box69\Downloads\cn_conversation.mp4"
# segments, info = model.transcribe(upload_audio_path, beam_size=10)

# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

# print(type(info))


DEVICE = 'cuda'
COMPUTE_TYPE = 'int8'


app = FastAPI()

model_instance = None

@app.on_event('startup')
async def startup_event():
    global model_instance
    model_instance = WhisperModel('large-v3', device=DEVICE, compute_type=COMPUTE_TYPE)


# def transcribe_generator(model_instance, file_path : str, beam_size : int):
#     segments, _ = 


@app.post('/audio/transcriptions')
async def transcribe_audio(file: UploadFile, model: str = "large-v3", beam_size : int = 10):
    # validate model name 
    if model not in ["tiny", "small", "medium", "large", "large-v2", "large-v3"]:
        raise HTTPException(status_code=400, detail=f"{model} not support.")
    
    # tempfile of upload audio
    try:
        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name
        # load model and transcribe
        try:
            model_instance = WhisperModel(model, device=DEVICE, compute_type=COMPUTE_TYPE)
            segments, info = model_instance.transcribe(tmp_file_path, beam_size=10)
            segments_list = [{"text": segment["text"], "start": segment["start"], "end": segment["end"]} for segment in segments]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"transcribe fail: {str(e)}")
        finally:
            # delete temp file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"upload audio file fail: {str(e)}")


    
    return {
        "language": info.get("language", "unknown"),  # 假设info字典包含语言信息
        "language_probability": info.get("language_probability", 0),  # 假设info字典包含语言概率信息
        "segments": segments_list
    }



