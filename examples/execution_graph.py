"""
OMNISCOPE Example: Execution Graph Construction & Querying
==========================================================

Demonstrates how trace events are assembled into a queryable
execution DAG, and how graph queries answer debugging questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any


# ---------------------------------------------------------------------------
# Lightweight Graph (no external dependencies)
# ---------------------------------------------------------------------------

@dataclass
class Node:
    event_id: str
    agent_id: str
    event_type: str
    timestamp: str
    latency_ms: float
    confidence: float | None = None
    risk_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    source: str  # event_id
    target: str  # event_id
    edge_type: str  # causal, data_dependency, reasoning_chain
    weight: float = 1.0


class ExecutionGraph:
    """
    In-memory DAG of trace events. In production this is Apache AGE (PostgreSQL).
    This example shows the query patterns the graph must support.
    """

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self._adjacency: dict[str, list[str]] = defaultdict(list)
        self._reverse_adjacency: dict[str, list[str]] = defaultdict(list)

    def add_node(self, node: Node) -> None:
        self.nodes[node.event_id] = node

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)
        self._adjacency[edge.source].append(edge.target)
        self._reverse_adjacency[edge.target].append(edge.source)

    # -------------------------------------------------------------------
    # QUERY: Causal chain (what caused this event?)
    # -------------------------------------------------------------------
    def causal_chain(self, event_id: str) -> list[str]:
        """
        Walks backward through causal edges to find all ancestors.

        Use case: "This output has a hallucination. What caused it?"
        Answer: The chain from the hallucinated output back through
                the writer, the vision agent's description, the
                retrieval results, etc.
        """
        visited = set()
        chain = []

        def dfs(eid: str):
            if eid in visited:
                return
            visited.add(eid)
            chain.append(eid)
            for parent in self._reverse_adjacency.get(eid, []):
                dfs(parent)

        dfs(event_id)
        return chain

    # -------------------------------------------------------------------
    # QUERY: Downstream impact (what did this event affect?)
    # -------------------------------------------------------------------
    def downstream_impact(self, event_id: str) -> list[str]:
        """
        Walks forward through edges to find all descendants.

        Use case: "If I change this retrieval step, what outputs are affected?"
        """
        visited = set()
        impacted = []

        def dfs(eid: str):
            if eid in visited:
                return
            visited.add(eid)
            impacted.append(eid)
            for child in self._adjacency.get(eid, []):
                dfs(child)

        dfs(event_id)
        return impacted

    # -------------------------------------------------------------------
    # QUERY: Critical path (longest latency chain)
    # -------------------------------------------------------------------
    def critical_path(self) -> list[str]:
        """
        Finds the longest-latency path through the DAG.

        Use case: "Why was this execution slow? Where's the bottleneck?"
        Returns the chain of events that determined total execution time.
        """
        # Topological sort + dynamic programming
        topo_order = self._topological_sort()
        dist: dict[str, float] = {eid: 0.0 for eid in self.nodes}
        predecessor: dict[str, str | None] = {eid: None for eid in self.nodes}

        for eid in topo_order:
            node = self.nodes[eid]
            current_dist = dist[eid] + node.latency_ms
            for child in self._adjacency.get(eid, []):
                if current_dist > dist[child]:
                    dist[child] = current_dist
                    predecessor[child] = eid

        # Find the end node with maximum distance
        end_node = max(dist, key=lambda k: dist[k])

        # Trace back
        path = []
        current = end_node
        while current is not None:
            path.append(current)
            current = predecessor[current]

        return list(reversed(path))

    # -------------------------------------------------------------------
    # QUERY: High-risk subgraph
    # -------------------------------------------------------------------
    def high_risk_subgraph(self, threshold: float = 0.5) -> list[str]:
        """
        Returns all events with risk_score above threshold,
        plus their causal ancestors.

        Use case: "Show me everything connected to high-risk events."
        This powers the Failure Risk Heatmap visualization.
        """
        high_risk = [
            eid for eid, node in self.nodes.items()
            if node.risk_score is not None and node.risk_score > threshold
        ]

        connected = set()
        for eid in high_risk:
            for ancestor in self.causal_chain(eid):
                connected.add(ancestor)

        return list(connected)

    # -------------------------------------------------------------------
    # QUERY: Agent interaction subgraph
    # -------------------------------------------------------------------
    def agent_interactions(self, agent_id: str) -> dict[str, list[Edge]]:
        """
        Returns all edges where agent_id is sender or receiver.

        Use case: "Show me all of Researcher agent's interactions."
        """
        result: dict[str, list[Edge]] = {"outgoing": [], "incoming": []}
        for edge in self.edges:
            source_node = self.nodes.get(edge.source)
            target_node = self.nodes.get(edge.target)
            if source_node and source_node.agent_id == agent_id:
                result["outgoing"].append(edge)
            if target_node and target_node.agent_id == agent_id:
                result["incoming"].append(edge)
        return result

    # -------------------------------------------------------------------
    # Internal: topological sort
    # -------------------------------------------------------------------
    def _topological_sort(self) -> list[str]:
        in_degree = defaultdict(int)
        for eid in self.nodes:
            in_degree[eid] = 0
        for edge in self.edges:
            in_degree[edge.target] += 1

        queue = [eid for eid in self.nodes if in_degree[eid] == 0]
        result = []
        while queue:
            eid = queue.pop(0)
            result.append(eid)
            for child in self._adjacency.get(eid, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return result


# ---------------------------------------------------------------------------
# Example: Build graph from the trace_log.json events
# ---------------------------------------------------------------------------

def build_example_graph() -> ExecutionGraph:
    """Constructs the execution graph from the product listing example."""
    g = ExecutionGraph()

    # Nodes (simplified from trace_log.json)
    events = [
        Node("evt-001", "orchestrator", "agent_decision", "T+0.00s", 120, 0.95, 0.12),
        Node("evt-002", "planner", "planning_step", "T+0.12s", 220, 0.92, 0.08),
        Node("evt-003", "vision", "image_transform", "T+0.36s", 50, None, 0.05),
        Node("evt-004", "vision", "modality_transition", "T+0.41s", 110, None, 0.05),
        Node("evt-005", "vision", "llm_call", "T+0.52s", 370, 0.93, 0.08),
        Node("evt-006", "writer", "llm_call", "T+0.91s", 930, 0.88, 0.18),
        Node("evt-007", "critic", "llm_call", "T+1.86s", 650, 0.85, 0.10),
        Node("evt-008", "system", "judge_evaluation", "T+2.91s", 400, None, 0.0),
        Node("evt-009", "writer", "llm_call", "T+3.12s", 680, 0.94, 0.05),
        Node("evt-010", "critic", "agent_decision", "T+3.98s", 220, 0.96, 0.03),
        Node("evt-011", "system", "judge_evaluation", "T+4.01s", 190, None, 0.0),
        Node("evt-012", "orchestrator", "agent_decision", "T+4.20s", 90, 0.97, 0.02),
    ]

    for node in events:
        g.add_node(node)

    # Edges (causal + data dependency)
    causal_edges = [
        ("evt-001", "evt-002"),
        ("evt-002", "evt-003"),
        ("evt-003", "evt-004"),
        ("evt-004", "evt-005"),
        ("evt-005", "evt-006"),
        ("evt-006", "evt-007"),
        ("evt-006", "evt-008"),
        ("evt-007", "evt-009"),
        ("evt-009", "evt-010"),
        ("evt-009", "evt-011"),
        ("evt-010", "evt-012"),
        ("evt-011", "evt-012"),
    ]

    for src, tgt in causal_edges:
        g.add_edge(Edge(src, tgt, "causal"))

    # Data dependencies (non-causal but influenced)
    g.add_edge(Edge("evt-005", "evt-006", "data_dependency"))
    g.add_edge(Edge("evt-005", "evt-007", "data_dependency"))
    g.add_edge(Edge("evt-006", "evt-009", "data_dependency"))

    return g


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    graph = build_example_graph()

    print("=== OMNISCOPE Execution Graph Demo ===\n")

    # Query 1: What caused the hallucination in evt-006?
    chain = graph.causal_chain("evt-006")
    print(f"Causal chain for writer's draft (evt-006):")
    for eid in chain:
        node = graph.nodes[eid]
        print(f"  {eid} [{node.agent_id}] {node.event_type} (latency: {node.latency_ms}ms)")

    print()

    # Query 2: What's the critical path?
    path = graph.critical_path()
    total_latency = sum(graph.nodes[eid].latency_ms for eid in path)
    print(f"Critical path ({total_latency}ms total):")
    for eid in path:
        node = graph.nodes[eid]
        print(f"  {eid} [{node.agent_id}] +{node.latency_ms}ms")

    print()

    # Query 3: High-risk subgraph
    risky = graph.high_risk_subgraph(threshold=0.15)
    print(f"High-risk subgraph (threshold 0.15):")
    for eid in risky:
        node = graph.nodes[eid]
        print(f"  {eid} [{node.agent_id}] risk={node.risk_score}")

    print()

    # Query 4: What does the vision agent's analysis affect?
    impact = graph.downstream_impact("evt-005")
    print(f"Downstream impact of vision analysis (evt-005):")
    for eid in impact:
        node = graph.nodes[eid]
        print(f"  {eid} [{node.agent_id}] {node.event_type}")
