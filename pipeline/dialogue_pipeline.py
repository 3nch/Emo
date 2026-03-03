import sys
import os
# 把项目根目录 E:/Emo 加入 Python 路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from prompt_builder import build_prompt
from inference.qwen_infer import main # 你已有的 Qwen3 推理代码
from emotion_detector.detect_emotion import detect_emotion
from safety.safety_detector import detect_safety_signal
from graphrag.graph_reasoner import GraphReasoner

def build_emotion_summary(traj):
    recent = traj.get_recent(3)
    if not recent:
        return ""

    lines = ["【用户近期情绪变化】"]
    for i, e in enumerate(recent, 1):
        lines.append(
            f"{i}. 情绪={e.emotion}, 强度={e.intensity:.2f}, 风险={e.risk}"
        )
    lines.append(f"总体趋势：{traj.trend()}")
    return "\n".join(lines)

class DialoguePipeline:
    def __init__(self):
        self.graph_reasoner = GraphReasoner()

    def run(self, user_text: str) -> str:
        emotion_result = detect_emotion(user_text)
        print(emotion_result)
        aLL_dicts = detect_safety_signal(user_text, emotion_result)
        print(aLL_dicts)
        state = {
            "user_text": user_text,
            "emotion": emotion_result["emotion"],
            "risk_level": aLL_dicts['risk_level'], #emotion_result["risk_level"],
            "safety": {
                "is_high_risk": aLL_dicts['is_high_risk'],
                "risk_type": aLL_dicts['risk_types'],
                "keywords": aLL_dicts['matched_keywords'],
            }
        }
        print(state)
        risk_level = state["risk_level"]

        if risk_level == "emergency":
            decision = self.graph_reasoner.force("P_EMERGENCY")
        elif risk_level == "高":
            decision = self.graph_reasoner.force("P_HIGH_RISK")
        elif risk_level == "中":
            decision = self.graph_reasoner.force("P_MEDIUM_RISK")
        else:
            decision = self.graph_reasoner.force("P_LOW_RISK")

        # ⭐ 关键一步：构造 Prompt
        prompt = build_prompt(
            user_text=user_text,
            emotion=state["emotion"],
            risk_level=state["risk_level"],
            decision=decision,
        )
        print(prompt)
        
        # ⭐ 喂给 Qwen3
        response = main(prompt)
        #response = generate_response(prompt)

        return response

if __name__ == "__main__":
    dialogue1 = DialoguePipeline()
    print("💬 情绪支持对话系统启动（输入 exit 退出）")

    while True:
        user_input = input("\n👤 用户：").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 对话结束")
            break

        reply = dialogue1.run(user_input)
        print(f"🤖 助手：{reply}")

