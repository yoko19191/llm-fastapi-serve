import httpx
import asyncio
from aiofiles import open as aio_open

async def stream_test():
    url = 'http://localhost:8000/v1/audio/transcriptions'
    audio_file_path = r'精神分析_何为白痴_为何这是个对精神病十分友好的时代.mp4'

    # 由于httpx不直接支持aiofiles的文件对象，需要先读取文件内容
    async with aio_open(audio_file_path, 'rb') as f:
        content = await f.read()

    files = {'file': ('filename.wav', content, 'audio/mp4')}
    data = {'model': 'large-v3', 'beam_size': '5', 'task': 'transcribe', 'stream': 'true'}

    # 修正：使用await等待post请求完成，而不是尝试用它作为async with的上下文管理器
    async with httpx.AsyncClient() as client:
        response = await client.post(url, files=files, data=data, timeout=None)
        # 现在response是一个响应对象，你可以按需处理它
        # 例如，逐行打印响应内容（对于流式传输）
        async for line in response.aiter_lines():
            print(line)

if __name__ == '__main__':
    asyncio.run(stream_test())
