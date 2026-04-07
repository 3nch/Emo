# safety/safety_detector.py
import re
import jieba
from typing import Dict, List, Tuple, Any

# ===================== 全局配置 =====================
USE_JIEBA = False   # 默认关闭分词（性能更好，必要时再打开）

# ===================== 风险词典 =====================
RISK_CONFIG: Dict[str, Dict[str, List[str]]] = {
    "emergency": {
        "self_harm": [
            "割腕", "跳楼", "服毒", "上吊", "跳河", "烧炭", "吞药", "撞墙",
            "买安眠药","找刀","自杀",
            "准备结束自己", "今晚就了结", "让自己解脱",
            "写遗书", "交代后事", "安排身后事","结束我的生命",
            "最后一次说", "再见了", "跟世界告别", "最后一次见面"
        ],
        "harm_others": [
            "买刀", "藏刀", "准备打人", "找棍子", "带凶器",
            "计划伤人", "找机会动手","杀了他","准备杀","复仇","整死他"
            "弄死他", "砍死他", "捅死他", "拉垫背的", "同归于尽","弄死"
        ]
    },
    "high": {
        "self_harm": [
            "伤害自己", "不想活了", "想死", "自我了结", "结束生命",
            "活不下去了", "撑不下去了", "一了百了","想去死",
            "活着没意义", "只想解脱", "了结此生","放弃生命",
            "放弃自己", "活够了", "生不如死", "不如死了算了"
        ],
        "harm_others": [
            "伤害别人", "想打人", "想砍人", "报复社会",
            "让他付出代价", "毁了他", "揍死他",
            "捅他", "杀了他", "伤害身边的人"
        ]
    },
    "medium": {
        "self_harm": [
            "想离开这个世界", "想永远睡过去", "想走了",
            "想消失在人海", "不想再面对了",
            "没什么可留恋的了", "人间不值得",
            "累了不想醒了", "活着太累了",
            "没有活下去的动力", "活受罪",
            "拖累家人", "摆烂到不想活"
        ],
        "harm_others": [
            "给点颜色看看", "让他后悔", "不会放过他",
            "讨个说法", "血债血偿", "以牙还牙",
            "找个人出气"
        ]
    },
    "low": {
        "emotion": [
            "控制不住自己", "无法控制", "失控",
            "情绪崩溃", "失去理智",
            "忍不住要动手", "想发泄",
            "什么都不在乎了", "豁出去了",
            "不想管后果了", "心态崩了",
            "情绪炸了", "压垮了"
        ]
    }
}

# ===================== 白名单 =====================
WHITELIST_KEYWORDS: List[str] = [
    "科普", "教程", "电影", "电视剧", "小说", "剧情",
    "案例", "讲解", "预防", "避免", "反对",
    "不要", "禁止", "警示", "教育", "宣传",
    "模拟", "演习", "测试", "举例", "说明", "分析", "讨论"
]

# ===================== 同义词扩展 =====================
SYNONYM_MAP: Dict[str, List[str]] = {
    "想死": ["想离世", "想归西", "想长眠", "想安息"],
    "割腕": ["划手腕", "割手", "划手"],
    "跳楼": ["跳窗", "跳阳台"],
    "打人": ["揍人", "动手打人", "动手"],
    "情绪崩溃": ["心态炸裂", "情绪失控", "精神崩溃"]
}

# ===================== 文本预处理 =====================
def _preprocess_text(text: str) -> str:
    if not text:
        return ""

    text = text.strip().lower()
    text = re.sub(r"[\s\/\\\,\.\!\?\;\:\'\"\+\-\*\_\(\)\[\]\{\}￥%@#&]", "", text)

    wrong_correct_map = {
        "想si": "想死",
        "活不下去le": "活不下去了",
        "自can": "自残",
        "tiao楼": "跳楼",
        "sha人": "杀人",
        "割wan": "割腕",
        "zou人": "揍人"
    }
    for wrong, correct in wrong_correct_map.items():
        text = text.replace(wrong, correct)

    for core, synonyms in SYNONYM_MAP.items():
        for s in synonyms:
            text = text.replace(s, core)

    return text

# ===================== 白名单过滤（按风险等级） =====================
def _filter_false_positive(text: str, risk_level: str) -> bool:
    has_whitelist = any(w in text for w in WHITELIST_KEYWORDS)

    if not has_whitelist:
        return False

    # 紧急 & 高风险：永不过滤
    if risk_level in ["emergency", "high"]:
        return False

    # 中低风险 + 白名单 → 过滤
    return True

# ===================== 主检测函数 =====================
def detect_safety_signal(text: str, emo_risk: dict) -> Dict[str, Any]:
    processed_text = _preprocess_text(text)
    if not processed_text:
        return {
            "is_high_risk": False,
            "risk_level": emo_risk['risk_level'],
            "risk_types": [],
            "matched_keywords": [],
#           "source": "rule_based"
        }

    if USE_JIEBA:
        segs = jieba.lcut(processed_text)
        text_for_match = processed_text + " " + " ".join(segs)
    else:
        text_for_match = processed_text

    risk_level = emo_risk["risk_level"]
    matched_keywords: List[str] = []
    risk_types: List[str] = []

    # 按优先级检测
    for level in ["emergency", "high", "medium", "low"]:
        level_matched = []
        level_types = []

        for type_key, keywords in RISK_CONFIG[level].items():
            for kw in keywords:
                if kw in text_for_match:
                    level_matched.append(kw)
                    level_types.append(type_key)

        if level_matched:
            risk_level = level
            matched_keywords = list(set(level_matched))
            risk_types = list(set(level_types))
            break

    # 白名单过滤
    if _filter_false_positive(processed_text, risk_level):
        return {
            "is_high_risk": False,
            "risk_level": emo_risk['risk_level'],
            "risk_types": [],
            "matched_keywords": [],
            "source": "rule_based"
        }

    is_high_risk = risk_level in ["emergency", "high"]

    return {
        "is_high_risk": is_high_risk,
        "risk_level": risk_level,
        "risk_types": risk_types,
        "matched_keywords": matched_keywords,
#       "source": "rule_based"
    }

# ===================== 兼容旧接口 =====================
def detect_safety_signal_simple(text: str) -> bool:
    return detect_safety_signal(text)["is_high_risk"]

# ===================== 测试 =====================
if __name__ == "__main__":
    tests = [
        "我买了安眠药，今晚就了结自己，再见了",
        "活不下去了，想死",
        "人间不值得，想永远睡过去",
        "科普：割腕的危害，如何预防自残行为",
        "我想si，准备tiao楼",
        "我要伤害别人",
        "电影里他想死，最后被救了"
    ]

    for t in tests:
        print("\n文本：", t)
        print(detect_safety_signal(t))
