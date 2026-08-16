"""Public types for temporal causal graphs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class CausalEdge:
    source: str
    target: str
    relation: str
    time: str
    weight: float = 1.0
    confidence: float = 1.0
    evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalEdge":
        return cls(**data)


@dataclass(slots=True)
class CausalPath:
    nodes: List[str]
    edges: List[CausalEdge]
    weight: float = 1.0
    confidence: float = 1.0
    evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": [e.to_dict() for e in self.edges],
            "weight": self.weight,
            "confidence": self.confidence,
            "evidence_ids": self.evidence_ids,
        }


@dataclass(slots=True)
class CausalExplanation:
    event: str
    paths: List[CausalPath]
    narrative: str = ""
    counterfactual: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event,
            "paths": [p.to_dict() for p in self.paths],
            "narrative": self.narrative,
            "counterfactual": self.counterfactual,
        }


@dataclass(slots=True)
class CausalQualityReport:
    edge_count: int = 0
    evidence_coverage: float = 0.0
    temporal_consistency: float = 1.0
    avg_confidence: float = 0.0
    cycle_rate: float = 0.0
    llm_judge_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

