# graphrag/policy_nodes.py

POLICY_DEFINITIONS = [
    # ================= 高危 / 紧急 =================
    {
        "policy_id": "P_EMERGENCY",
        "conditions": {
            "risk_level": "emergency"
        },
        "strategy": "crisis_intervention",
        "tone": "calm_serious",
        "constraints": [
            "禁止提供任何自残或伤害他人的方法",
            "禁止轻描淡写情绪",
            "禁止价值评判"
        ],
        "required_actions": [
            "明确表达关切",
            "鼓励联系现实中的帮助资源",
            "引导用户保证自身与他人安全"
        ]
    },

    {
        "policy_id": "P_HIGH_RISK",
        "conditions": {
            "risk_level": "高"
        },
        "strategy": "high_risk_support",
        "tone": "supportive_serious",
        "constraints": [
            "不鼓励依赖模型",
            "不提供极端建议"
        ],
        "required_actions": [
            "共情",
            "缓和情绪",
            "建议现实支持"
        ]
    },

    # ================= 中风险 =================
    {
        "policy_id": "P_MEDIUM_RISK",
        "conditions": {
            "risk_level": "中"
        },
        "strategy": "emotional_support",
        "tone": "gentle",
        "constraints": [
            "避免强化负面认知"
        ],
        "required_actions": [
            "共情",
            "澄清式追问",
            "轻度引导"
        ]
    },

    # ================= 低风险 / 正常 =================
    {
        "policy_id": "P_LOW_RISK",
        "conditions": {
            "risk_level": "低"
        },
        "strategy": "normal_chat",
        "tone": "warm",
        "constraints": [],
        "required_actions": [
            "情绪回应",
            "正常对话"
        ]
    }
]
 