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
            "content": "你是一名情绪安抚助手，目标是理解和陪伴，不做评判。"
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
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    with torch.no_grad():
        model.generate(
            **inputs,
            streamer = streamer,
            max_new_tokens = 256,
            do_sample = True,
            temperature = 0.7,
            top_p = 0.9,
            repetition_penalty = 1.1,
            eos_token_id = tokenizer.eos_token_id,
        )


if __name__ == "__main__":
    main()
