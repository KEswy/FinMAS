"""Lightweight dynamic event knowledge graph.

The default backend is in-memory. Neo4j remains an optional future backend;
the interface does not require it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass(slots=True)
class KGNode:
    node_id: str
    layer: str
    name: str
    node_type: str
    attributes: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class KGEdge:
    src: str
    dst: str
    relation: str
    weight: float = 1.0
    confidence: float = 1.0
    ttl_days: Optional[float] = None
    created_at_days: float = 0.0

    def effective_weight(self, as_of_days: float) -> float:
        w = self.weight * self.confidence
        if self.ttl_days is not None:
            age = max(as_of_days - self.created_at_days, 0.0)
            half_life = max(self.ttl_days, 1e-6)
            w *= math.exp(-math.log(2) * age / half_life)
        return w


@dataclass(slots=True)
class SubGraph:
    nodes: List[KGNode]
    edges: List[KGEdge]
    as_of_days: float = 0.0

    def summary(self) -> str:
        lines = ["[知识图谱子图摘要]"]
        for n in self.nodes:
            lines.append(f"  节点 [{n.layer}] {n.name} ({n.node_type})")
        for e in self.edges:
            lines.append(
                f"  {e.src} --[{e.relation}, w={e.effective_weight(self.as_of_days):.3f}]--> {e.dst}"
            )
        return "\n".join(lines)


class KnowledgeGraph:
    """Event/industry/company/asset graph built from local event metadata."""

    LAYERS = ("social", "industry", "company", "asset")

    def __init__(self) -> None:
        self.nodes: Dict[str, KGNode] = {}
        self.edges: List[KGEdge] = []

    def add_node(self, node: KGNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: KGEdge) -> None:
        self.edges.append(edge)

    def build_from_event_library(self, event_csv: str = "data/events/event_library.csv") -> None:
        import pandas as pd

        try:
            df = pd.read_csv(event_csv)
        except FileNotFoundError:
            return
        df["event_date"] = df["event_date"].astype(str).str[:10]
        for _, row in df.iterrows():
            event_id = f"event_{row['event_date']}_{row['event_type']}"
            self.add_node(
                KGNode(event_id, "social", str(row["event_name"]), "policy_event")
            )
            industries = self._parse_industries(row.get("affected_industries"))
            for code in industries:
                ind_id = f"industry_{code}"
                self.add_node(KGNode(ind_id, "industry", str(code), "sector"))
                self.add_edge(
                    KGEdge(event_id, ind_id, "affects", weight=0.8,
                           confidence=0.8, ttl_days=60)
                )

    @staticmethod
    def _parse_industries(value: object) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(x).strip() for x in value if str(x).strip()]
        text = str(value)
        return [x.strip() for x in re.split(r"[,，;；\s]+", text) if x.strip()]

    def get_subgraph(self, entity_names: Sequence[str], max_hops: int = 3,
                     edge_threshold: float = 0.05, as_of_days: float = 0.0) -> SubGraph:
        names = [str(n) for n in entity_names]
        seeds = {
            nid
            for nid, node in self.nodes.items()
            if any(name in node.name or name == node.node_id for name in names)
        }
        visited = set(seeds)
        frontier = set(seeds)
        for _ in range(max_hops):
            nxt = set()
            for edge in self.edges:
                if edge.effective_weight(as_of_days) < edge_threshold:
                    continue
                if edge.src in frontier and edge.dst not in visited:
                    nxt.add(edge.dst)
                if edge.dst in frontier and edge.src not in visited:
                    nxt.add(edge.src)
            visited |= nxt
            frontier = nxt
            if not nxt:
                break
        nodes = [self.nodes[nid] for nid in visited if nid in self.nodes]
        edges = [
            e for e in self.edges
            if e.src in visited and e.dst in visited
            and e.effective_weight(as_of_days) >= edge_threshold
        ]
        return SubGraph(nodes=nodes, edges=edges, as_of_days=as_of_days)

    def propagate_impact(self, seed_id: str, initial_strength: float = 1.0,
                         as_of_days: float = 0.0) -> Dict[str, float]:
        scores = {seed_id: initial_strength}
        frontier = [(seed_id, initial_strength)]
        visited = {seed_id}
        while frontier:
            src, strength = frontier.pop(0)
            src_node = self.nodes.get(src)
            if src_node is None:
                continue
            src_layer = self.LAYERS.index(src_node.layer) if src_node.layer in self.LAYERS else -1
            for edge in self.edges:
                if edge.src != src or edge.dst in visited:
                    continue
                dst_node = self.nodes.get(edge.dst)
                if dst_node is None:
                    continue
                dst_layer = self.LAYERS.index(dst_node.layer) if dst_node.layer in self.LAYERS else -1
                if dst_layer < src_layer:
                    continue
                new_strength = strength * edge.effective_weight(as_of_days)
                if new_strength < 0.05:
                    continue
                scores[edge.dst] = scores.get(edge.dst, 0.0) + new_strength
                visited.add(edge.dst)
                frontier.append((edge.dst, new_strength))
        return scores

