"""Interpretable direction fusion and selective prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..schemas import Direction, DirectionDecision, EventInput, SignalBundle
from .calibration import TemporalCalibrator

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SelectivePolicy:
    thresholds: Dict[str, float] = field(default_factory=dict)
    default_threshold: float = 0.56

    def threshold_for(self, event_type: str) -> float:
        return float(self.thresholds.get(event_type, self.default_threshold))

    @classmethod
    def from_validation(
        cls,
        probs: Sequence[float],
        labels: Sequence[int],
        event_types: Sequence[str],
        coverage_target: float = 0.65,
    ) -> "SelectivePolicy":
        arr = np.asarray(probs, dtype=float)
        lab = np.asarray(labels, dtype=int)
        ets = [str(e) for e in event_types]
        out: Dict[str, float] = {}
        for et in sorted(set(ets)):
            mask = np.array([e == et for e in ets])
            if int(mask.sum()) < 5:
                continue
            p = arr[mask]
            y = lab[mask]
            # Find the highest probability threshold retaining the target coverage.
            quantile = max(0.0, 1.0 - coverage_target)
            out[et] = float(np.quantile(p, quantile))
        return cls(thresholds=out)


class DirectionFusion:
    def __init__(self, calibrator: Optional[TemporalCalibrator] = None,
                 policy: Optional[SelectivePolicy] = None) -> None:
        self.calibrator = calibrator or TemporalCalibrator()
        self.policy = policy or SelectivePolicy()
        self.feature_names: List[str] = [
            "disagreement",
            "froth",
            "irf_prior",
            "llm_bull_bear",
            "llm_net",
            "ts_prob_up",
            "ts_signal",
            "valuation_signal",
        ]

    def vector_array(self, bundle: SignalBundle) -> np.ndarray:
        vec = self._vector(bundle)
        return np.nan_to_num(
            np.asarray([vec.get(name, 0.0) for name in self.feature_names], dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _vector(self, bundle: SignalBundle) -> Dict[str, float]:
        factors = bundle.factor_vector()
        def num(value: object, default: float = 0.0) -> float:
            try:
                out = float(value)
            except Exception:
                return default
            return default if not np.isfinite(out) else out

        llm_net = num(np.sum([v for k, v in factors.items() if k.startswith("llm_")]))
        positive = num(np.sum([max(v, 0.0) for v in factors.values()]))
        negative = num(np.sum([max(-v, 0.0) for v in factors.values()]))
        ts_prob = num(factors.get("ts_prob_up", 0.5), 0.5)
        ts_signal = num(factors.get("ts_signal", 0.0))
        valuation = num(factors.get("valuation_signal", 0.0))
        froth = num(factors.get("froth", 0.0))
        irf = num(factors.get("irf_prior", 0.0))
        disagreement = num(factors.get("disagreement", 0.0))
        return {
            "llm_net": llm_net,
            "llm_bull_bear": positive - negative,
            "ts_prob_up": ts_prob,
            "ts_signal": ts_signal,
            "valuation_signal": valuation,
            "froth": froth,
            "irf_prior": irf,
            "disagreement": disagreement,
        }

    def raw_score(self, bundle: SignalBundle) -> float:
        vec = self._vector(bundle)
        score = (
            0.55 * vec["ts_prob_up"]
            + 0.15 * np.tanh(vec["llm_bull_bear"] / 1.5)
            + 0.10 * np.tanh(vec["valuation_signal"])
            + 0.10 * np.tanh(vec["froth"])
            + 0.10 * np.tanh(vec["irf_prior"])
            - 0.15 * np.clip(vec["disagreement"], 0.0, 1.0)
        )
        return float(np.clip(score, 0.0, 1.0))

    def fit_from_past(self, features: np.ndarray, labels: np.ndarray,
                      event_types: Sequence[str]) -> "DirectionFusion":
        self.calibrator.fit(features, labels)
        if len(features) > 0:
            raw = self.calibrator.predict_proba(features)
            self.policy = SelectivePolicy.from_validation(raw, labels, event_types)
        return self

    def decide(self, bundle: SignalBundle, horizon: int = 5) -> DirectionDecision:
        vec = self._vector(bundle)
        raw = self.raw_score(bundle)
        if self.calibrator._model is not None:
            prob_up = float(self.calibrator.predict_proba(self.vector_array(bundle))[0])
        else:
            prob_up = raw

        event = bundle.event
        threshold = self.policy.threshold_for(event.event_type)
        confidence = float(abs(prob_up - 0.5) * 2.0)
        disagreement = float(vec["disagreement"])
        evidence_count = len(bundle.evidence.chunks) if bundle.evidence else 0

        abstain = False
        reasons = []
        if confidence < abs(threshold - 0.5) * 2.0:
            abstain = True
            reasons.append("low_confidence")
        if disagreement > 0.55:
            abstain = True
            reasons.append("high_disagreement")
        if evidence_count == 0 and bundle.llm_factors:
            reasons.append("no_retrieved_evidence")

        direction = Direction.FLAT.value
        if not abstain:
            direction = Direction.UP.value if prob_up >= 0.5 else Direction.DOWN.value

        evidence_ids = []
        if bundle.evidence:
            evidence_ids = [c.chunk_id for c in bundle.evidence.chunks[:10]]

        return DirectionDecision(
            event_date=event.event_date,
            industry_code=event.industry_code,
            horizon=int(horizon),
            prob_up=prob_up,
            final_direction=direction,
            confidence=confidence,
            abstain=abstain,
            abstain_reason=";".join(reasons) if reasons else "",
            evidence_ids=evidence_ids,
            reason="calibrated fusion",
            factors=vec,
        )
