from openai import OpenAI
import os
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
# ============ DashScope OpenAI 兼容配置 ============
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

from collections import deque

def main(response: str):

    messages = [
        {
            "role": "system",
            "content": (
                "你是一名情绪支持型对话助手。\n"
                "你的目标是：\n"
                "1️⃣ 真诚共情用户的感受\n"
                "2️⃣ 用温和、耐心、不评判的方式回应\n"
                "3️⃣ 适当展开解释，帮助用户理解情绪来源\n"
                "4️⃣ 通过提问或建议，引导用户缓和情绪\n\n"
                "请使用完整自然的段落进行回应，"
                "不要过于简短，也不要只用一句话结束。"
            )
        },
        {
            "role": "user",
            "content": response
        }
    ]

    completion = client.chat.completions.create(
        model="qwen-max",   # 可换 qwen-turbo / qwen-max
        messages=messages,
        temperature=0.8,
        top_p=0.95,
        max_tokens=768,
    )

    reply = completion.choices[0].message.content
    # print(reply)
    return reply


if __name__ == "__main__":
    main("我最近真的很累，感觉什么都提不起兴趣")