import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

# 路径配置
DATA_DIR = r"E:\Emo\data\EATD-Corpus\EATD-Corpus"
OUTPUT_DIR = r"E:\Emo\data\processed"
SEGMENT_LEN = 10 

def process_eatd_v2():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")

    manifest = []
    # 1. 直接获取目录下所有以 't_' 开头的文件夹
    p_folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f)) and f.startswith('t_')]
    
    print(f"发现 {len(p_folders)} 个受访者文件夹。开始处理...")

    for p_id in tqdm(p_folders):
        p_folder = os.path.join(DATA_DIR, p_id)
        
        # 2. 读取标签 (SDS Score)
        label_path = os.path.join(p_folder, "new_label.txt")
        if not os.path.exists(label_path):
            continue
            
        with open(label_path, 'r') as f:
            try:
                sds_score = float(f.read().strip())
            except: continue
        
        binary_label = 1 if sds_score >= 53 else 0

        # 3. 遍历该文件夹下三种情绪的音频
        for mood in ['positive', 'neutral', 'negative']:
            # 优先选择 _out.wav
            audio_file = os.path.join(p_folder, f"{mood}_out.wav")
            if not os.path.exists(audio_file):
                audio_file = os.path.join(p_folder, f"{mood}.wav")
            
            if not os.path.exists(audio_file):
                continue

            # 4. 加载、采样、切割
            try:
                # librosa加载速度可能较慢，但对抑郁症检测的采样率对齐很关键
                y, sr = librosa.load(audio_file, sr=16000)
                samples_per_seg = SEGMENT_LEN * sr
                
                for i, start in enumerate(range(0, len(y), samples_per_seg)):
                    chunk = y[start : start + samples_per_seg]
                    if len(chunk) < 3 * sr: continue # 片段太短则跳过
                    
                    chunk_name = f"{p_id}_{mood}_{i}.wav"
                    save_path = os.path.join(OUTPUT_DIR, chunk_name)
                    sf.write(save_path, chunk, sr)
                    
                    manifest.append({
                        "path": save_path,
                        "label": binary_label,
                        "score": sds_score,
                        "mood": mood
                    })
            except Exception as e:
                print(f"处理 {audio_file} 失败: {e}")

    # 保存清单
    df = pd.DataFrame(manifest)
    df.to_csv("eatd_manifest.csv", index=False)
    print(f"\n处理完成！生成片段总数: {len(manifest)}")
    print(f"清单已保存至: {os.getcwd()}\eatd_manifest.csv")

if __name__ == "__main__":
    process_eatd_v2()