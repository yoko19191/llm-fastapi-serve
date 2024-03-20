from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from tempfile import NamedTemporaryFile
from faster_whisper import WhisperModel
import os 
import torch 


import logging

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

app = FastAPI()



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRANSCRIBE_MODEL_SIZE = "large-v3"
TRANSCRIBE_COMPUTE_TYPE = "int8_float16" 


transcribe_model = None
llm_model = None



def transcribe_generator(audio_file_path, beam_size : int, task : str):
    global transcribe_model
    
    segments, info = transcribe_model.transcribe(audio_file_path, beam_size=beam_size, vad_filter=True, task=task)

    print(f"Detected language: {info.language} with probability: {info.language_probability}")

    for segment in segments:
        yield f"data: [{segment.start} -> {segment.end}] {segment.text}\n\n"



@app.on_event("startup")
async def startup_event():
    print(f"Initializing transcribe model....device={DEVICE}, model_size={TRANSCRIBE_MODEL_SIZE}, compute_type={TRANSCRIBE_COMPUTE_TYPE}")
    global transcribe_model
    transcribe_model = WhisperModel(TRANSCRIBE_MODEL_SIZE, device=DEVICE, compute_type=TRANSCRIBE_COMPUTE_TYPE)



@app.post("/v1/audio/transcriptions")
async def transcribe_audio(file: UploadFile, model: str = "large-v3", beam_size: int = 5, task: str = "transcribe", stream: bool = False):
    # validate transcribe model
    if model not in ["tiny", "small", "medium", "large", "large-v2", "large-v3"]:
        raise HTTPException(status_code=400, detail=f"{model} not supported.")
    # validate task
    if task not in ["transcribe", "translate"]:
        raise HTTPException(status_code=400, detail=f"{task} not supported.")

    # deal with temp audio file 
    try:
        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name

        # transcribe happens here
        try:
            assert os.path.exists(tmp_file_path), f"File does not exist: {tmp_file_path}"  # make sure temp file exists

            segments, info = transcribe_model.transcribe(tmp_file_path, beam_size=beam_size, vad_filter=True, task=task)

            if stream is not True:
                transcriptions = [segment.text for segment in segments]
                full_transcription = " ".join(transcriptions)

                # return result
                return {
                    "language": info.language,
                    "language_probability": info.language_probability,
                    "text": full_transcription
                }
            else:
                response_generator = transcribe_generator(tmp_file_path, beam_size=beam_size, task=task)
                return StreamingResponse(response_generator, media_type="text/event-stream")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"transcribe failed: {str(e)}")
        finally:
            # delete temp file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"upload audio file failed: {str(e)}")
    
