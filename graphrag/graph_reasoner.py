# graphrag/graph_reasoner.py
from typing import Dict, Any
from graphrag.graph_builder import build_decision_graph


class GraphReasoner:
    def __init__(self):
        self.graph = build_decision_graph()

    def reason(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        state = {
          emotion,
          risk_level,
          safety_level
        }
        """
        candidates = self.graph.get_next_nodes("START", state)

        if not candidates:
            return self._default_strategy()

        # 优先级：emergency > high > medium > low
        selected = candidates[0]
        data = selected.data

        return {
            "strategy": data["strategy"],
            "tone": data["tone"],
            "constraints": data["constraints"],
            "required_actions": data["required_actions"]
        }

    @staticmethod
    def _default_strategy():
        return {
            "strategy": "normal_chat",
            "tone": "neutral",
            "constraints": [],
            "required_actions": []
        }
    
    def force(self, policy_id: str) -> Dict[str, Any]:
        node = self.graph.nodes[policy_id]
        data = node.data
        return {
            "strategy": data["strategy"],
            "tone": data["tone"],
            "constraints": data["constraints"],
            "required_actions": data["required_actions"]
        }