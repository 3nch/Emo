import json
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

MODEL_PATH = "./emotion_qwen25_lora_1"

model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_PATH,
    max_seq_length = 512,
    load_in_4bit = True,
)

FastLanguageModel.for_inference(model)

def detect_emotion(text):
    prompt = (
        "你是一个情绪检测模型。\n"
        "请根据用户输入，判断其情绪状态和风险等级。\n"
        "只能输出JSON，格式如下：\n"
        "{\"emotion\":\"情绪\",\"risk_level\":\"风险\"}\n\n"
        f"用户输入：{text}\n"
        "输出："
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        temperature=0.0,
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    json_str = result.split("输出：")[-1].strip()

    return json.loads(json_str)

# # 测试
# tests = [
#     "我最近总觉得自己什么都做不好",
#     "最近压力好大，晚上睡不着",
#     "今天心情还可以",
#     "事情永远有新的，永远做不完",
#     "我的人生好失败哎，活着也没意思",
#     "好烦啊，今天又被领导骂惨了"
# ]

# for t in tests:
#     print(t)
#     print(detect_emotion(t))
#     print("-" * 40)
