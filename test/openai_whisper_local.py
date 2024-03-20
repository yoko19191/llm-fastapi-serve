# https://github.com/openai/whisper

import argparse
import whisper 
import time
import os

MODEL_NAME = "large-v3"

def transcribe_audio(audio_file_path):
    # Load the Whisper model
    model = whisper.load_model(MODEL_NAME)
    print(f"model device: {model.device}")
    
    # Load the audio file and adjust its length
    audio = whisper.load_audio(audio_file_path)
    audio = whisper.pad_or_trim(audio)
    
    # Generate log-Mel spectrogram
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    # Detect the spoken language in the audio
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    print(f"Detected language: {detected_language}")
    
    # Decode the audio to text
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    
    # Print the recognized text
    print(result.text)

# 创建解析器
parser = argparse.ArgumentParser(description="将音频文件转录成文本。")
# 添加 `-F` 选项用于指定音频文件路径
parser.add_argument('-F', '--file', type=str, required=True, help="音频文件的路径。")
# 解析命令行参数
args = parser.parse_args()
audio_path = args.file


time_start = time.time()
result = transcribe_audio(audio_path)
time_end = time.time()


print("转录完成，用时：", time_end - time_start, "秒。")
print("转录文本：", result.text)

# 构建输出文件名（与输入文件同名，扩展名为.txt），但仅在当前目录下保存
output_file_name = os.path.splitext(os.path.basename(audio_path))[0] + ".txt"

# 保存转录文本
with open(output_file_name, "w", encoding="utf-8") as text_file:
    text_file.write(result.text)

print(f"转录文本已保存到本地文件：{output_file_name}。")



