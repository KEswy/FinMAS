"""Causal explanation and counterfactual helpers."""

from __future__ import annotations

from typing import List, Optional

from .graph import TemporalCausalGraph
from .schemas import CausalExplanation, CausalPath


class CausalExplainer:
    def __init__(self, graph: TemporalCausalGraph) -> None:
        self.graph = graph

    def explain(self, event: str, target: str = "market",
                top_k: int = 5, as_of: Optional[str] = None) -> CausalExplanation:
        paths = self.graph.shortest_causal_paths(event, target, top_k=top_k, as_of=as_of)
        narrative = self._narrative(event, target, paths)
        return CausalExplanation(event=event, paths=paths, narrative=narrative)

    def _narrative(self, source: str, target: str, paths: List[CausalPath]) -> str:
        if not paths:
            return f"未找到从 {source} 到 {target} 的因果路径。"
        parts = []
        for i, path in enumerate(paths[:3], 1):
            parts.append(
                f"路径{i}: {' → '.join(path.nodes)}，置信度 {path.confidence:.2f}"
            )
        return "；".join(parts)

    def counterfactual(self, event: str, target: str,
                       perturbation: dict) -> dict:
        original = self.explain(event, target)
        # A simple deterministic perturbation: flip event-target edge relation.
        counter_paths = []
        for path in original.paths:
            if path.edges:
                first = path.edges[0]
                counter_paths.append(
                    CausalPath(
                        nodes=list(path.nodes),
                        edges=[
                            CausalEdge(
                                source=first.source,
                                target=first.target,
                                relation="contradicts",
                                time=first.time,
                                weight=first.weight,
                                confidence=first.confidence,
                                evidence_ids=first.evidence_ids,
                            )
                        ]
                        + path.edges[1:],
                        weight=path.weight,
                        confidence=path.confidence,
                        evidence_ids=path.evidence_ids,
                    )
                )
        return {
            "perturbation": perturbation,
            "original": original.to_dict(),
            "counterfactual_narrative": self._narrative(event, target, counter_paths),
        }

