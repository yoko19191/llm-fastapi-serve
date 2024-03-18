from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from tempfile import NamedTemporaryFile
from faster_whisper import WhisperModel
import os 