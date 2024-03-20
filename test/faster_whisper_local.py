# https://github.com/SYSTRAN/faster-whisper

import argparse
from faster_whisper import WhisperModel
import time
import os
import torch

# 定义设备，模型大小和计算类型
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SIZE = "large-v3"
COMPUTE_TYPE = "int8_float16" 

print(f"MODEL_SIZE: {MODEL_SIZE}, device: {DEVICE}, COMPUTE_TYPE: {COMPUTE_TYPE}")

# 定义转录音频的函数，这次作为生成器使用
def transcribe_audio(audio_file_path):
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    segments, info = model.transcribe(audio_file_path, beam_size=5, vad_filter=True)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        yield segment.start, segment.end, segment.text

# 解析命令行参数
parser = argparse.ArgumentParser(description="将音频文件转录成文本。")
parser.add_argument('-F', '--file', type=str, required=True, help="音频文件的路径。")
args = parser.parse_args()

# 测量转录时间
time_start = time.time()

# 调用转录音频函数并逐段处理结果
with open(os.path.splitext(os.path.basename(args.file))[0] + ".txt", "w", encoding="utf-8") as text_file:
    for start, end, text in transcribe_audio(args.file):
        text_file.write(f"[{start:.2f}s -> {end:.2f}s] {text}\n")
        print(f"[{start:.2f}s -> {end:.2f}s] {text}")

time_end = time.time()

print(f"转录完成，用时：{time_end - time_start}秒。")
print(f"转录文本已保存到本地文件：{os.path.splitext(os.path.basename(args.file))[0] + '.txt'}。")
