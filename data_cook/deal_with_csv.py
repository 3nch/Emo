import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

# ===================== 配置 =====================
RAW_ROOT = r"E:\Emo\data\daicwoz"
OUT_ROOT = r"E:\Emo\clips"
SR = 16000        # 统一采样率
MIN_DURATION = 0.3  # 最短语音（秒），太短的直接丢
# ===============================================


def process_one_subject(subj_dir):
    """
    subj_dir: E:\\Emo\\data\\daicwoz\\300_P
    """
    pid = subj_dir.replace("_P", "")
    wav_path = os.path.join(RAW_ROOT, subj_dir, f"{pid}_AUDIO.wav")
    csv_path = os.path.join(RAW_ROOT, subj_dir, f"{pid}_TRANSCRIPT.csv")

    if not os.path.exists(wav_path) or not os.path.exists(csv_path):
        print(f"[跳过] {pid} 缺少 AUDIO 或 TRANSCRIPT")
        return

    # 输出目录
    out_dir = os.path.join(OUT_ROOT, pid)
    os.makedirs(out_dir, exist_ok=True)

    # 读取 transcript（TAB 分隔）
    df = pd.read_csv(csv_path, sep="\t")

    # 只保留 Participant
    df = df[df["speaker"] == "Participant"].copy()

    if len(df) == 0:
        print(f"[警告] {pid} 没有 Participant 语音")
        return

    # 读音频
    y, sr = librosa.load(wav_path, sr=SR)

    saved = 0
    for idx, row in df.iterrows():
        start = float(row["start_time"])
        end = float(row["stop_time"])

        if end - start < MIN_DURATION:
            continue

        s = int(start * sr)
        e = int(end * sr)

        if s >= len(y) or e <= s:
            continue

        clip = y[s:e]

        out_path = os.path.join(
            out_dir,
            f"{pid}_{saved:04d}.wav"
        )

        sf.write(out_path, clip, sr)
        saved += 1

    print(f"[完成] {pid} 切出 {saved} 条语音")


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    subj_dirs = [
        d for d in os.listdir(RAW_ROOT)
        if d.endswith("_P") and os.path.isdir(os.path.join(RAW_ROOT, d))
    ]

    print(f"发现被试数：{len(subj_dirs)}")

    for d in tqdm(subj_dirs):
        process_one_subject(d)


if __name__ == "__main__":
    main()
