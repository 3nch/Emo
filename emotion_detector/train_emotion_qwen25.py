import os
import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import os
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "emotion_train.jsonl")

    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

    # 1. 加载模型（Unsloth 会自动用 bf16）
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=512,
        load_in_4bit=True,
        dtype=None,   # 让 Unsloth 自己决定
    )

    # 2. LoRA（判别任务，配置是合理的）
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        use_gradient_checkpointing=True,
    )

    # 3. 数据集（⚠ 不要多进程）
    dataset = load_dataset(
        "json",
        data_files=DATA_PATH,
    )

    # 4. prompt 构造（判别式，OK）
    def format_func(batch):
        texts = []
        for user_input, output in zip(batch["input"], batch["output"]):
            text = (
                "你是一个情绪检测模型。\n"
                "请根据用户输入，判断其情绪状态和风险等级。\n"
                "只能输出JSON，格式如下：\n"
                "{\"emotion\":\"情绪\",\"risk_level\":\"风险\"}\n\n"
                f"用户输入：{user_input}\n"
                "输出："
                f"{output}"
            )
            texts.append(text)
        return texts



    # 5. Trainer（关键修改在这里）
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        formatting_func=format_func,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=3,

            # ⭐ 关键：bf16 / fp16
            fp16=False,
            bf16=True,

            logging_steps=10,
            output_dir="./emotion_qwen25_lora",
            save_strategy="epoch",
            report_to="none",
            torch_compile=False,
            # ⭐ 防止 Windows 再炸
            dataloader_num_workers=0,
        ),
    )

    trainer.train()

    model.save_pretrained("./emotion_qwen25_lora_1")
    tokenizer.save_pretrained("./emotion_qwen25_lora_1")


if __name__ == "__main__":
    main()
