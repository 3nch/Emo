import torch
import librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

# --- 配置 ---
MODEL_PATH = r"E:\Emo\best_depression_model" # 指向你保存的最佳模型文件夹
AUDIO_FILE = r"E:\Emo\data\processed\t_1_positive_0.wav"          # 替换成你想测试的音频路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict():
    # 1. 加载模型和特征提取器
    print("正在加载模型...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    # 2. 处理音频
    speech, _ = librosa.load(AUDIO_FILE, sr=16000)
    inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(DEVICE)

    # 3. 推理
    with torch.no_grad():
        logits = model(input_values).logits
        
    # 4. 结果处理
    probs = torch.nn.functional.softmax(logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][prediction].item()

    labels = {0: "健康 (Normal)", 1: "抑郁 (Depressed)"}
    print(f"\n--- 检测结果 ---")
    print(f"音频文件: {AUDIO_FILE}")
    print(f"检测结论: {labels[prediction]}")
    print(f"置信度: {confidence:.2%}")

if __name__ == "__main__":
    predict()