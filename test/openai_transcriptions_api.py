# https://platform.openai.com/docs/api-reference/audio/createTranscription

import argparse
from openai import OpenAI
from dotenv import load_dotenv
import time
import os

load_dotenv()


# 创建解析器
parser = argparse.ArgumentParser(description="将音频文件转录成文本。")
# 添加 `-F` 选项用于指定音频文件路径
parser.add_argument('-F', '--file', type=str, required=True, help="音频文件的路径。")
# 解析命令行参数
args = parser.parse_args()
audio_path = args.file

client = OpenAI()

time_start = time.time()


transcription_response = client.audio.transcriptions.create(
    model="whisper-1",
    file=open(audio_path, "rb")
)

time_end = time.time()

print("转录完成，用时：", time_end - time_start, "秒。")
print("转录文本：", transcription_response.text)

# 构建输出文件名（与输入文件同名，扩展名为.txt），但仅在当前目录下保存
output_file_name = os.path.splitext(os.path.basename(audio_path))[0] + ".txt"

# 保存转录文本
with open(output_file_name, "w", encoding="utf-8") as text_file:
    text_file.write(transcription_response.text)

print(f"转录文本已保存到本地文件：{output_file_name}。")



