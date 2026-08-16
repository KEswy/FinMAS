"""Graph state and debate verdict contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..schemas import DirectionDecision, EventInput, EvidencePackage, FactorSignal


@dataclass(slots=True)
class DebateVerdict:
    consensus_factors: List[FactorSignal] = field(default_factory=list)
    disagreement: float = 0.0
    contested: bool = False
    evidence_ids: List[str] = field(default_factory=list)
    rounds: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consensus_factors": [asdict(f) for f in self.consensus_factors],
            "disagreement": self.disagreement,
            "contested": self.contested,
            "evidence_ids": self.evidence_ids,
            "rounds": self.rounds,
        }


@dataclass(slots=True)
class GraphState:
    event: EventInput
    evidence: Optional[EvidencePackage] = None
    mechanism_signals: List[FactorSignal] = field(default_factory=list)
    debate_transcript: List[Dict[str, str]] = field(default_factory=list)
    judge_verdict: Optional[DebateVerdict] = None
    time_series: Dict[str, Any] = field(default_factory=dict)
    valuation: Dict[str, float] = field(default_factory=dict)
    sentiment: Dict[str, float] = field(default_factory=dict)
    irf_prior: Dict[str, float] = field(default_factory=dict)
    disagreement: float = 0.0
    decision: Optional[DirectionDecision] = None
    risk: Dict[str, Any] = field(default_factory=dict)
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event,
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "mechanism_signals": [asdict(f) for f in self.mechanism_signals],
            "debate_transcript": self.debate_transcript,
            "judge_verdict": self.judge_verdict.to_dict() if self.judge_verdict else None,
            "time_series": self.time_series,
            "valuation": self.valuation,
            "sentiment": self.sentiment,
            "irf_prior": self.irf_prior,
            "disagreement": self.disagreement,
            "decision": self.decision.to_dict() if self.decision else None,
            "risk": self.risk,
            "trace": self.trace,
        }

