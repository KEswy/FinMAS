"""Public contracts for continuous market simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class MarketTick:
    date: str
    market_return: float
    industry_returns: Dict[str, float] = field(default_factory=dict)
    capital_flow: Dict[str, float] = field(default_factory=dict)
    triggered_event: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentOpinion:
    agent: str
    opinion: str
    confidence: float
    evidence_ids: List[str] = field(default_factory=list)
    disagreement: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CausalPath:
    source: str
    target: str
    relation: str
    weight: float
    evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TimelineEntry:
    date: str
    tick: MarketTick
    opinions: List[AgentOpinion] = field(default_factory=list)
    causal_paths: List[CausalPath] = field(default_factory=list)
    narrative: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "tick": self.tick.to_dict(),
            "opinions": [o.to_dict() for o in self.opinions],
            "causal_paths": [p.to_dict() for p in self.causal_paths],
            "narrative": self.narrative,
        }


@dataclass(slots=True)
class CounterfactualResult:
    perturbation: Dict[str, Any]
    original_narrative: str
    counterfactual_narrative: str
    changed_paths: List[CausalPath] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perturbation": self.perturbation,
            "original_narrative": self.original_narrative,
            "counterfactual_narrative": self.counterfactual_narrative,
            "changed_paths": [p.to_dict() for p in self.changed_paths],
        }


@dataclass(slots=True)
class SimulationState:
    timeline: List[TimelineEntry] = field(default_factory=list)
    counterfactuals: List[CounterfactualResult] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timeline": [e.to_dict() for e in self.timeline],
            "counterfactuals": [c.to_dict() for c in self.counterfactuals],
            "trace": self.trace,
        }

