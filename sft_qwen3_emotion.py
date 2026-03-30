from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import os

os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

# ─────────────────── 环境变量（你原来的保留）───────────────────
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# ─────────────────── 主逻辑一定要放 main ────────────────────
def main():

    max_seq_length = 4096
    dtype = None
    load_in_4bit = True

    # ───────────── 加载模型（关键：device_map）────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = r"E:\EMO\Qwen3",   # 你的本地模型路径
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        device_map = {"": 0},           # ⭐ 必须
    )

    # ───────────── LoRA 设置 ─────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha = 32,
        lora_dropout = 0.05,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # ───────────── formatting_func（现在 tokenizer 已存在）────────────
    def formatting_prompts_func(example):
        texts = []

        # 单条 or batch 兼容
        if isinstance(example["messages"], list) and isinstance(example["messages"][0], dict):
            messages_list = [example["messages"]]
        else:
            messages_list = example["messages"]

        for messages in messages_list:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize = False,
                add_generation_prompt = False,
            )
            if not text.endswith(tokenizer.eos_token):
                text += tokenizer.eos_token
            texts.append(text)

        return texts


    # ───────────── 加载数据集 ─────────────
    dataset = load_dataset(
        "json",
        data_files = "data/fine_tune_data.jsonl",
        split = "train",
    )

    # ───────────── Trainer ─────────────
    import torch._dynamo
    torch._dynamo.disable()
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        formatting_func = formatting_prompts_func,
        max_seq_length = max_seq_length,
        dataset_num_proc = 1,     # ⭐ Windows 必须 = 1
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            num_train_epochs = 2,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs/qwen3-emotion-lora",
            report_to = "none",
            torch_compile = False,
        ),
    )

    trainer.train()

    # ───────────── 保存 ─────────────
    model.save_pretrained("qwen3-emotion-lora")
    tokenizer.save_pretrained("qwen3-emotion-lora")


# ─────────────────── Windows 必须 ────────────────────
if __name__ == "__main__":
    main()
