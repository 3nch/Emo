import torch
import pandas as pd
import librosa
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# --- 配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "facebook/wav2vec2-large-xlsr-53"
CSV_PATH = r"E:\EMO\data\processed\eatd_manifest.csv"
BATCH_SIZE = 2 
ACCUMULATION_STEPS = 4
EPOCHS = 40
LR = 2e-5  # 略微调高，帮助跳出局部最优

class EATDDataset(Dataset):
    def __init__(self, df, feature_extractor):
        self.df = df
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['label']
        try:
            speech, _ = librosa.load(path, sr=16000)
            inputs = self.feature_extractor(speech, sampling_rate=16000, return_tensors="pt").input_values
            return {"input_values": inputs.squeeze(0), "labels": torch.tensor(label, dtype=torch.long)}
        except: return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    return {
        "input_values": nn.utils.rnn.pad_sequence([item["input_values"] for item in batch], batch_first=True),
        "labels": torch.stack([item["labels"] for item in batch])
    }

def train():
    # 1. 加载并过滤数据：只保留 negative，这是检测抑郁的关键
    df = pd.read_csv(CSV_PATH)
    print(f"原始数据量: {len(df)}")
    df = df[df['mood'] == 'negative'].reset_index(drop=True)
    print(f"过滤后（仅保留negative）数据量: {len(df)}")

    # 2. 划分数据集
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    # 3. 构建过采样器 (WeightedRandomSampler) - 解决 F1=0 的核心
    target = train_df['label'].values
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(np.array([weight[t] for t in target])).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # 4. 初始化模型与工具
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2).to(DEVICE)
    
    # 冻结前期，后期全量微调
    model.freeze_feature_encoder()

    train_loader = DataLoader(EATDDataset(train_df, feature_extractor), batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(EATDDataset(val_df, feature_extractor), batch_size=BATCH_SIZE, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler('cuda')
    
    # 因为已经使用了Sampler，Loss函数可以不用带weight
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    print(f"\n[开始训练] 目标显卡: {torch.cuda.get_device_name(0)}")

    for epoch in range(EPOCHS):
        # 第10轮起解冻所有层，进行全量微调
        if epoch == 10:
            print("\n--- [通知] 解冻全量参数进行深度微调 ---")
            for param in model.parameters():
                param.requires_grad = True

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for i, batch in enumerate(pbar):
            if batch is None: continue
            with autocast('cuda'):
                logits = model(batch["input_values"].to(DEVICE)).logits
                loss = criterion(logits, batch["labels"].to(DEVICE)) / ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # 验证
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                logits = model(batch["input_values"].to(DEVICE)).logits
                all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                all_labels.extend(batch["labels"].numpy())

        f1 = f1_score(all_labels, all_preds, zero_division=0)
        acc = accuracy_score(all_labels, all_preds)
        print(f"\n[Epoch {epoch+1}] Val Acc: {acc:.4f} | Val F1: {f1:.4f}")
        
        if f1 > best_f1 and f1 > 0:
            best_f1 = f1
            model.save_pretrained("./best_depression_model")
            print(f">>> F1 突破记录: {f1:.4f}，模型已保存。")

if __name__ == "__main__":
    train()