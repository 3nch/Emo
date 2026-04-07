# graphrag/graph_schema.py
from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class GraphNode:
    node_id: str
    node_type: str          # state / policy / action
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    condition: Dict[str, Any]  # 条件，如 {"risk_level": "high"}


class DecisionGraph:
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []

    def add_node(self, node: GraphNode):
        self.nodes[node.node_id] = node

    def add_edge(self, edge: GraphEdge):
        self.edges.append(edge)

    def get_next_nodes(self, current_node_id: str, state: Dict[str, Any]) -> List[GraphNode]:
        results = []
        for edge in self.edges:
            if edge.source != current_node_id:
                continue
            if self._match_condition(edge.condition, state):
                results.append(self.nodes[edge.target])
        return results

    @staticmethod
    def _match_condition(condition: Dict[str, Any], state: Dict[str, Any]) -> bool:
        for k, v in condition.items():
            if state.get(k) != v:
                return False
        return True
