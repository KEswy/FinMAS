"""Public data contracts for FinMAS v5.

The contracts intentionally keep the direction decision separate from LLM
output. LLM agents emit structured evidence, not final direction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class Direction(str, Enum):
    UP = "+"
    DOWN = "-"
    FLAT = "0"


@dataclass(slots=True)
class EventInput:
    """One leak-safe event/industry scenario."""

    event_date: str
    event_text: str
    event_type: str
    industry_code: str
    event_id: str = ""
    external_features: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> Tuple[str, str]:
        return str(self.event_date)[:10], str(self.industry_code)


@dataclass(slots=True)
class EvidenceChunk:
    chunk_id: str
    text: str
    source: str
    pub_time: float
    score: float = 0.0
    entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvidencePackage:
    query: str
    chunks: List[EvidenceChunk] = field(default_factory=list)
    subgraph_summary: str = ""
    total_tokens_est: int = 0
    passed_time_firewall: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "chunks": [c.to_dict() for c in self.chunks],
            "subgraph_summary": self.subgraph_summary,
            "total_tokens_est": self.total_tokens_est,
            "passed_time_firewall": self.passed_time_firewall,
        }


@dataclass(slots=True)
class FactorSignal:
    """A signed, bounded factor with provenance and confidence."""

    name: str
    value: float
    confidence: float
    direction: str = "+"
    source: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def signed_value(self) -> float:
        sign = -1.0 if self.direction == Direction.DOWN else 1.0
        return sign * float(self.value) * float(self.confidence)


@dataclass(slots=True)
class SignalBundle:
    """All evidence available to the calibrated fusion layer."""

    event: EventInput
    llm_factors: List[FactorSignal] = field(default_factory=list)
    time_series: Dict[str, Any] = field(default_factory=dict)
    valuation: Optional[Dict[str, float]] = None
    sentiment: Optional[Dict[str, float]] = None
    irf_prior: Optional[Dict[str, float]] = None
    disagreement: float = 0.0
    evidence: Optional[EvidencePackage] = None

    def factor_vector(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for f in self.llm_factors:
            out[f.name] = f.signed_value()
        if self.valuation:
            out.setdefault("valuation_signal", float(self.valuation.get("signal", 0.0)))
        if self.sentiment:
            out.setdefault("froth", float(self.sentiment.get("froth", 0.0)))
        if self.irf_prior:
            out.setdefault("irf_prior", float(self.irf_prior.get("signed_strength", 0.0)))
        out.setdefault("ts_signal", float(self.time_series.get("signal", 0.0)))
        out.setdefault("ts_prob_up", float(self.time_series.get("prob_up", 0.5)))
        out.setdefault("disagreement", float(self.disagreement))
        return out


@dataclass(slots=True)
class DirectionDecision:
    event_date: str
    industry_code: str
    horizon: int
    prob_up: float
    final_direction: str
    confidence: float
    abstain: bool = False
    abstain_reason: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    reason: str = ""
    factors: Dict[str, float] = field(default_factory=dict)
    risk: Optional["RiskStats"] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_date": self.event_date,
            "industry_code": self.industry_code,
            "horizon": self.horizon,
            "prob_up": round(float(self.prob_up), 6),
            "final_direction": self.final_direction,
            "confidence": round(float(self.confidence), 6),
            "abstain": bool(self.abstain),
            "abstain_reason": self.abstain_reason,
            "evidence_ids": list(self.evidence_ids),
            "reason": self.reason,
            "factors": {k: round(float(v), 6) for k, v in self.factors.items()},
            "risk": self.risk.to_dict() if self.risk else None,
        }


@dataclass(slots=True)
class RiskStats:
    horizon: int
    var: float
    expected_shortfall: float
    expected_mdd: float
    lower_interval: float
    upper_interval: float
    tail_prob: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon,
            "var": round(self.var, 8),
            "expected_shortfall": round(self.expected_shortfall, 8),
            "expected_mdd": round(self.expected_mdd, 8),
            "lower_interval": round(self.lower_interval, 8),
            "upper_interval": round(self.upper_interval, 8),
            "tail_prob": round(self.tail_prob, 8),
        }


@dataclass(slots=True)
class RiskReport:
    event_date: str
    industry_code: str
    horizons: Dict[int, RiskStats] = field(default_factory=dict)
    stress: Dict[str, Any] = field(default_factory=dict)
    attribution: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_date": self.event_date,
            "industry_code": self.industry_code,
            "horizons": {str(k): v.to_dict() for k, v in self.horizons.items()},
            "stress": self.stress,
            "attribution": self.attribution,
            "warnings": self.warnings,
        }

