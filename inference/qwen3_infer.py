import os
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main(response):

    # ───────────── 路径配置 ─────────────
    BASE_MODEL_PATH = r"E:\EMO\Qwen3"              # 基础模型路径
    LORA_PATH = r"qwen3-emotion-lora"              # 你训练好的 LoRA
    max_seq_length = 4096

    # ───────────── 加载模型 + LoRA ─────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL_PATH,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
        device_map = {"": 0},
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,   # 和训练时一致
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha = 32,
        lora_dropout = 0.0,
        bias = "none",
    )

    # 加载 LoRA 权重
    adapter_name = "my_lora"
    model.load_adapter(LORA_PATH,adapter_name=adapter_name)

    model.eval()

    # ───────────── 构造测试对话 ─────────────
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

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors = "pt"
    ).to("cuda")

    # ───────────── 推理参数 ─────────────
    streamer = TextStreamer(tokenizer, skip_prompt=True,skip_special_tokens=True,)

    with torch.no_grad():
        model.generate(
            **inputs,
            streamer = streamer,
            max_new_tokens = 768,
            do_sample = True,
            temperature = 0.8,
            top_p = 0.95,
            repetition_penalty = 1.1,
            #eos_token_id = tokenizer.eos_token_id,
        )


if __name__ == "__main__":
    main()
