"""
Transmission contract (v4 Phase 1)
==================================

Upgrades the reasoning→numeric interface from a FLAT scalar to a STRUCTURED,
Shenwan-industry-keyed transmission map — without changing behavior.

Why this exists
---------------
v3 collapsed all mechanism reasoning into ``to_exogenous_factors() ->
Dict[str,float]`` keyed by an arbitrary LLM-chosen entity string, then summed
to a single ``net_signal`` and multiplied by ONE ``(event_type, industry)``
sensitivity coefficient (mechanism_agent.py:33 -> main.py:396-402 -> 539-551).
There is no industry structure in the mechanism output; the industry only
enters via the sensitivity lookup.

The transmission-analysis skill (Phase 3) and the risk-control objective
(Phase 4) both need a per-industry structure — one policy shock fans out to
many sectors, each with its own direction / magnitude / confidence / path.
So the contract becomes a ``TransmissionMap``. To de-risk the migration, this
map ``project_to_scalar()``s back to exactly the v3 ``net_signal`` for the
single industry under prediction, so Phase 1 is behavior-preserving: the
walk-forward accuracy must equal Phase 0 within noise.

Migration
---------
- ``MechanismChain.to_transmission_map()`` becomes the source of truth.
- ``project_to_scalar(map, industry_code)`` reproduces the old net_signal.
- Batch scripts reading ``result["exog_factors"]`` get a back-compat flat view
  via ``TransmissionMap.as_exog_factors()`` for one release.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math


# Magnitude vocabulary shared with the v3 mechanism chain (mechanism_agent.py:39)
_MAG_MAP = {"高": 0.9, "中": 0.5, "低": 0.2, "未知": 0.0}


@dataclass
class IndustryCell:
    """One (event → industry) transmission cell."""
    industry_code: str                       # Shenwan first-tier, e.g. "801780"
    direction: int = 0                       # +1 / -1 / 0 (unknown)
    magnitude: float = 0.0                   # [0,1]
    confidence: float = 0.0                  # [0,1]
    path: List[str] = field(default_factory=list)   # macro-var transmission trail
    irf_prior: Optional[float] = None        # signed DSGE-IRF prior (Phase 3), or None
    entity: str = ""                         # original free-text entity (provenance)

    def signed_score(self) -> float:
        """v3-equivalent per-cell factor = direction × magnitude × confidence."""
        return self.direction * self.magnitude * self.confidence


@dataclass
class TransmissionMap:
    """Structured, Shenwan-keyed transmission of one event across industries.

    Two coexisting views:
      - ``raw_factors``: the EXACT v3 ``to_exogenous_factors()`` dict (entity-keyed,
        per-step, last-write-wins on duplicate entity). This is the source of the
        behavior-preserving scalar — ``project_to_scalar()`` sums it and must equal
        v3's ``net_signal`` byte-for-byte.
      - ``cells``: an industry-code-keyed AGGREGATED view for the Phase 3 transmission
        skill and Phase 4 risk head. Derived, never feeds the scalar.
    """
    event_date: str = ""
    event_type: str = ""
    cells: Dict[str, IndustryCell] = field(default_factory=dict)
    raw_factors: Dict[str, float] = field(default_factory=dict)

    # ── construction ────────────────────────────────────────────
    def add_cell(self, cell: IndustryCell) -> None:
        self.cells[cell.industry_code] = cell

    @classmethod
    def from_mechanism_chain(cls, chain, industry_hint: str = "",
                             event_date: str = "", event_type: str = "") -> "TransmissionMap":
        """Build from a v3 MechanismChain, behavior-preservingly.

        ``raw_factors`` reproduces v3 ``to_exogenous_factors()`` EXACTLY
        (mechanism_agent.py:33-42): entity-keyed, ``direction = +1 iff
        impact_direction=='+' else -1`` (note: unknown/'-'/'' ALL map to -1 in
        v3), magnitude via _MAG_MAP, last-write-wins on duplicate entity. This
        is what feeds the scalar, so numbers match Phase 0.

        ``cells`` is a separate industry-code-keyed aggregation for new skills;
        it uses a finer 3-way direction (+1/-1/0) and does NOT affect the scalar.
        """
        tm = cls(event_date=event_date, event_type=event_type)
        for step in getattr(chain, "steps", []):
            entity = step.get("entity", "")
            # --- v3-exact raw factor (feeds scalar) ---
            v3_dir = 1.0 if step.get("impact_direction") == "+" else -1.0
            magnitude = _MAG_MAP.get(step.get("impact_magnitude", "未知"), 0.0)
            conf = float(step.get("confidence", 0.5))
            tm.raw_factors[entity] = v3_dir * magnitude * conf   # last-write-wins

            # --- aggregated cell view (feeds Phase 3/4, not the scalar) ---
            code = step.get("industry_code") or industry_hint or entity
            fine_dir = 1 if step.get("impact_direction") == "+" else (
                -1 if step.get("impact_direction") == "-" else 0)
            if code in tm.cells:
                c = tm.cells[code]
                c.magnitude = max(c.magnitude, magnitude)
                c.confidence = max(c.confidence, conf)
                if c.direction == 0:
                    c.direction = fine_dir
            else:
                tm.add_cell(IndustryCell(
                    industry_code=code, direction=fine_dir,
                    magnitude=magnitude, confidence=conf, entity=entity,
                ))
        return tm

    # ── back-compat views (EXACT v3 semantics, drive the scalar) ─────
    def as_exog_factors(self) -> Dict[str, float]:
        """The exact v3 ``exog_factors`` dict (entity-keyed). Drop-in for any
        code that consumed ``chain.to_exogenous_factors()``."""
        return dict(self.raw_factors)

    def project_to_scalar(self, industry_code: str = "") -> float:
        """Reproduce v3 ``net_signal`` = ``sum(exog_factors.values())``
        (main.py:399 net_signal_inject; :538-541 pos_sum-neg_sum == full sum).
        Byte-identical to v3 when industry_code is omitted.

        Passing industry_code restricts to that industry's aggregated cell
        (the Phase 3+ per-industry path; NOT used by the behavior-preserving
        Phase 1 shim)."""
        if industry_code:
            c = self.cells.get(industry_code)
            return c.signed_score() if c else 0.0
        return sum(self.raw_factors.values())

    def event_strength(self) -> float:
        """v3 ``event_strength`` = max-abs raw factor (main.py:397-398)."""
        vals = list(self.raw_factors.values())
        return float(max(vals, key=abs)) if vals else 0.0

    # ── structured signals for Phase 3/4 ────────────────────────
    def net_direction(self) -> int:
        s = self.project_to_scalar()
        return 1 if s > 0 else (-1 if s < 0 else 0)

    def dispersion(self) -> float:
        """Cross-industry disagreement — a risk feature (Phase 4).
        Std of per-cell signed scores; high dispersion = conflicting sector
        reactions = higher predictive uncertainty."""
        scores = [c.signed_score() for c in self.cells.values()]
        if len(scores) < 2:
            return 0.0
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        return math.sqrt(var)

    def coverage(self) -> int:
        """How many industries the map actually resolved a direction for."""
        return sum(1 for c in self.cells.values() if c.direction != 0)

    def to_dict(self) -> dict:
        return {
            "event_date": self.event_date,
            "event_type": self.event_type,
            "cells": {
                code: {
                    "direction": c.direction, "magnitude": c.magnitude,
                    "confidence": c.confidence, "path": c.path,
                    "irf_prior": c.irf_prior, "entity": c.entity,
                } for code, c in self.cells.items()
            },
            "net_direction": self.net_direction(),
            "dispersion": self.dispersion(),
            "coverage": self.coverage(),
        }
