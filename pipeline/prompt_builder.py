# pipeline/prompt_builder.py

def build_prompt(
    user_text: str,
    emotion: str,
    risk_level: str,
    decision: dict,
    history: list = None,
) -> str:
    """
    将 pipeline 决策结果转成最终喂给 Qwen3 的 Prompt
    history: 对话历史列表，每项为 {'role': 'user'/'assistant', 'content': '...'}
    """

    # ===== 1. System Role =====
    system_role = (
        "你是一名情绪支持型对话助手，目标是理解用户感受、认真有效有温度的安抚用户的情绪"
        "稳定情绪，并在必要时引导用户寻求现实中的支持。"
        "你不进行评判，不提供极端或危险建议。"
    )

    # ===== 2. Strategy 描述 =====
    strategy_map = {
        "high_risk_support": "当前用户处于高风险状态，需要谨慎、严肃且温和地回应。",
        "medium_risk_support": "当前用户存在一定心理压力，需要共情和引导。",
        "normal_chat": "当前用户状态较为稳定，可进行正常对话。"
    }

    strategy_text = strategy_map.get(
        decision["strategy"],
        "请以支持性的方式回应用户。"
    )

    # ===== 3. Tone =====
    tone_text = f"回应语气要求：{decision['tone']}。"

    # ===== 4. Constraints =====
    constraints_text = ""
    if decision.get("constraints"):
        constraints_text = "请严格遵守以下限制：\n"
        for c in decision["constraints"]:
            constraints_text += f"- {c}\n"

    # ===== 5. Required Actions =====
    actions_text = ""
    if decision.get("required_actions"):
        actions_text = "回应中需要包含以下要点：\n"
        for a in decision["required_actions"]:
            actions_text += f"- {a}\n"

    # ===== 6. User Context =====
    user_context = (
        f"【用户情绪判断】\n"
        f"- 情绪：{emotion}\n"
        f"- 风险等级：{risk_level}\n\n"
        f"【用户原始输入】\n{user_text}"
    )

    # ===== 7. 构建最终 Prompt（包含历史对话）=====
    # 历史对话格式
    history_text = ""
    if history and len(history) > 0:
        history_text = "【对话历史】\n"
        for h in history[-6:]:  # 保留最近6轮对话
            role = "用户" if h.get('role') == 'user' else "助手"
            history_text += f"{role}：{h.get('content', '')}\n"
        history_text += "\n"

    final_prompt = f"""
{system_role}

{strategy_text}
{tone_text}

{constraints_text}

{actions_text}

{history_text}

{user_context}

请基于以上信息，给出合适的中文回复。
""".strip()

    return final_prompt
