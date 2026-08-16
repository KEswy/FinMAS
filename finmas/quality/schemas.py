"""Contracts for module-quality reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class Threshold:
    name: str
    operator: str  # ">=" | "<=" | "=="
    value: float

    def check(self, observed: float) -> bool:
        if self.operator == ">=":
            return float(observed) >= self.value
        if self.operator == "<=":
            return float(observed) <= self.value
        if self.operator == "==":
            return float(observed) == self.value
        raise ValueError(f"unknown operator: {self.operator}")


@dataclass(slots=True)
class QualityMetric:
    name: str
    value: float
    passed: bool
    threshold: float | None = None
    operator: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": round(float(self.value), 6),
            "passed": bool(self.passed),
            "threshold": self.threshold,
            "operator": self.operator,
        }


@dataclass(slots=True)
class QualityReport:
    module: str
    metrics: List[QualityMetric] = field(default_factory=list)
    passed: bool = False
    n_samples: int = 0
    notes: List[str] = field(default_factory=list)

    def finalize(self) -> "QualityReport":
        self.passed = bool(self.metrics) and all(m.passed for m in self.metrics)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "metrics": [m.to_dict() for m in self.metrics],
            "passed": self.passed,
            "n_samples": self.n_samples,
            "notes": self.notes,
        }

