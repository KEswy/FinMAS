"""FinMAS v5: a firewall-safe multi-agent financial analysis system."""

from .schemas import (
    Direction,
    DirectionDecision,
    EventInput,
    EvidenceChunk,
    EvidencePackage,
    SignalBundle,
    RiskStats,
    RiskReport,
)

__all__ = [
    "Direction",
    "DirectionDecision",
    "EventInput",
    "EvidenceChunk",
    "EvidencePackage",
    "SignalBundle",
    "RiskStats",
    "RiskReport",
]

__version__ = "5.0.0"

