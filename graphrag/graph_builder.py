# graphrag/graph_builder.py
from graphrag.graph_schema import DecisionGraph, GraphNode, GraphEdge
from graphrag.policy_nodes import POLICY_DEFINITIONS


def build_decision_graph() -> DecisionGraph:
    graph = DecisionGraph()

    # 起始节点
    graph.add_node(GraphNode(
        node_id="START",
        node_type="state"
    ))

    # 策略节点
    for policy in POLICY_DEFINITIONS:
        node = GraphNode(
            node_id=policy["policy_id"],
            node_type="policy",
            data=policy
        )
        graph.add_node(node)

        graph.add_edge(GraphEdge(
            source="START",
            target=policy["policy_id"],
            condition=policy["conditions"]
        ))

    return graph
