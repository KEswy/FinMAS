"""
Transmission skill (v4 Phase 3)
===============================

Blends a transmission-economics IRF prior (data/irf_priors.json, built +
adversarially verified by the irf-prior-build workflow) into the LLM
mechanism chain's directional signal, via a single PRE-REGISTERED fixed
weight (soft blend, no fitting — the overfitting guard for n=168).

Motivation: Phase-0 walk-forward showed negative-CAR events at only 50%.
The LLM reads policy text and over-commits to "利好" (positive). The IRF prior
encodes where transmission theory + history say a nominally-bullish policy is
actually bearish for a sector (e.g. rate cut → bank NIM compression). Soft-
blending pulls the directional signal toward the IRF where they disagree.

Firewall compliance (critical):
  - The IRF *direction* is a global transmission-theory constant (no leakage).
  - The IRF *strength* in the json was calibrated on the FULL sample. At
    prediction time for event t we RE-SCALE strength using only <t history
    (recalibrate_strength), so no future CAR leaks into the magnitude.

Blend (all on the bounded directional-signal scale [-1,1]):
    llm_sig  = tanh(net_signal / TANH_SCALE)         # LLM direction, squashed
    irf_sig  = direction * strength_wf                # IRF signed prior, <t-scaled
    blend    = (1-λ)·llm_sig + λ·irf_sig              # soft mix
    net_out  = blend · |net_signal| / max(|llm_sig|, ε)   # map back to net_signal scale,
                                                          # preserving LLM magnitude envelope
Only the SIGN/relative-direction is nudged; the overall correction magnitude
stays governed by the existing SCALE/clip pipeline in main.py.

Disabled by default (USE_IRF=0) so Phase-0/1 behavior is reproducible and the
skill is cleanly ablatable (on/off comparison is the Phase-3 evaluation).
"""
from __future__ import annotations

import os
import json
import math
import datetime as _dt
from typing import Optional, Tuple

import pandas as pd

# Pre-registered hyperparameters (NOT fitted — frozen before evaluation).
BLEND_LAMBDA = float(os.environ.get("IRF_LAMBDA", "0.35"))   # weight on IRF prior
TANH_SCALE = float(os.environ.get("IRF_TANH_SCALE", "2.0"))  # net_signal→[-1,1] squash
_EPS = 1e-6


def _to_date(x) -> _dt.date:
    if isinstance(x, _dt.date) and not isinstance(x, _dt.datetime):
        return x
    return _dt.datetime.strptime(str(x)[:10], "%Y-%m-%d").date()


class TransmissionSkill:
    """Loads the IRF prior and blends it into the directional signal."""

    def __init__(self, irf_path: str = "data/irf_priors.json",
                 car_csv: str = os.environ.get("CAR_CSV", "data/processed/car_results.csv"),
                 blend_lambda: float = BLEND_LAMBDA):
        self.blend_lambda = blend_lambda
        self.car_csv = car_csv
        try:
            with open(irf_path, encoding="utf-8") as f:
                self._priors = json.load(f).get("priors", {})
        except FileNotFoundError:
            self._priors = {}
        self._car5: Optional[pd.DataFrame] = None

    # ── IRF lookup ───────────────────────────────────────────────
    def irf_cell(self, event_type: str, industry_code: str) -> Optional[dict]:
        return self._priors.get(event_type, {}).get(str(industry_code))

    def irf_signed(self, event_type: str, industry_code: str) -> Tuple[int, float]:
        """(direction, strength) from the global prior; (0,0) if no cell."""
        c = self.irf_cell(event_type, industry_code)
        if not c:
            return 0, 0.0
        return int(c.get("direction", 0)), float(c.get("strength", 0.0))

    # ── firewall-safe strength recalibration ─────────────────────
    def _car5_df(self) -> pd.DataFrame:
        if self._car5 is None:
            df = pd.read_csv(self.car_csv)
            df = df[df["window"] == 5].copy()
            df["industry_code"] = df["industry_code"].astype(str)
            df["_d"] = df["event_date"].apply(_to_date)
            self._car5 = df
        return self._car5

    def recalibrate_strength(self, event_type: str, industry_code: str,
                             event_dt, base_strength: float) -> float:
        """Re-scale |strength| by the <t historical sign-consistency of this
        (event_type, industry) cell, so magnitude never uses future data.
        Direction (sign of base_strength) is preserved from theory.

        scale = 2·|P_pos(<t) − 0.5| in [0,1]: strong when history is one-sided,
        →0 when history is a coin flip. n<3 (<t) → shrink hard (×0.3)."""
        if base_strength == 0.0:
            return 0.0
        try:
            d = self._car5_df()
            t = _to_date(event_dt)
            sub = d[(d["event_type"] == event_type) &
                    (d["industry_code"] == str(industry_code)) &
                    (d["_d"] < t)]
            n = len(sub)
            if n == 0:
                return 0.0                         # cold start: no <t history → IRF abstains
            p_pos = float((sub["CAR"] >= 0).mean())
            consistency = 2.0 * abs(p_pos - 0.5)   # [0,1]
            scale = consistency if n >= 3 else 0.3 * consistency
        except Exception:
            scale = 1.0
        sign = 1.0 if base_strength > 0 else -1.0
        return sign * abs(base_strength) * scale

    # ── pre-t conviction (leak-free) ─────────────────────────────
    def pre_t_conviction(self, event_type: str, industry_code: str,
                         event_dt) -> Tuple[float, int]:
        """Strictly pre-t sign-consistency of this cell: (conviction, n).
        conviction = 2|P_pos(<t)-0.5| in [0,1]; n<3 -> caller abstains.
        Uses ONLY event_date<t rows, so it is leak-free."""
        try:
            d = self._car5_df()
            t = _to_date(event_dt)
            sub = d[(d["event_type"] == event_type) &
                    (d["industry_code"] == str(industry_code)) &
                    (d["_d"] < t)]
            n = len(sub)
            if n == 0:
                return 0.0, 0
            p_pos = float((sub["CAR"] >= 0).mean())
            return 2.0 * abs(p_pos - 0.5), n
        except Exception:
            return 0.0, 0

    # ── the blend ────────────────────────────────────────────────
    def blend_net_signal(self, net_signal: float, event_type: str,
                         industry_code: str, event_dt,
                         walk_forward: bool = True) -> Tuple[float, dict]:
        """Return (blended_net_signal, info). If no IRF cell, returns net_signal
        unchanged. walk_forward controls whether strength is <t-recalibrated.

        IRF_MODE=leakfree (v2, reviewer redesign): direction is the FROZEN
        transmission-theory sign (never re-derived from realized CAR -> closes
        the 5d leak); influence is conviction-gated by strictly pre-t
        sign-consistency (thin/mixed history -> abstain), so a HIGH-conviction
        bearish cell can actually flip the LLM's over-optimistic sign, while
        generic shrinkage (which never changes sign) cannot. Targets direction,
        not point-MAE (which zero-correction already wins)."""
        mode = os.environ.get("IRF_MODE", "v1").strip().lower()
        direction, base_strength = self.irf_signed(event_type, industry_code)
        if direction == 0 and base_strength == 0.0:
            return net_signal, {"irf": "none"}

        llm_sig = math.tanh(net_signal / TANH_SCALE)          # [-1,1]
        envelope = abs(net_signal) if abs(net_signal) > _EPS else 1.0

        if mode == "leakfree":
            # v2: theory direction + pre-t conviction gate. No full-sample strength.
            # 5d fix: only trust cells where theory and history AGREED at build time
            # (verified_agree=true => direction IS the theory sign). Cells whose
            # sign was overridden by full-sample history (verified_agree=false) are
            # abstained, so NO realized-CAR-derived direction can enter.
            cell = self.irf_cell(event_type, industry_code) or {}
            if not bool(cell.get("verified_agree", True)):
                return net_signal, {"irf": "abstain_leaky_dir", "direction": direction}
            conviction, n = self.pre_t_conviction(event_type, industry_code, event_dt)
            if direction == 0 or n < 3 or conviction <= 0.0:
                return net_signal, {"irf": "abstain", "direction": direction,
                                    "n_pre_t": n, "conviction": round(conviction, 3)}
            lam_max = float(os.environ.get("IRF_LAMBDA_MAX", "0.6"))
            lam_eff = lam_max * conviction                    # gated: decisive only when pre-t one-sided
            sent_info = {}
            # ── Sentiment gate (USE_SENTIMENT=1): froth modulates trust in the
            # theory prior. gate = 1 + BETA*froth*direction, so hot froth vs a
            # bearish cell (froth>0,dir<0) SUPPRESSES it (retail overrides
            # fundamentals); cold froth vs bearish AMPLIFIES it. froth is pre-t,
            # market-level -> leak-free. BETA fixed, not tuned.
            if os.environ.get("USE_SENTIMENT", "0") == "1":
                from agents.sentiment_agent import SentimentSkill
                if not hasattr(self, "_sent"):
                    self._sent = SentimentSkill()
                froth, sent_info = self._sent.froth(event_dt)
                beta = float(os.environ.get("SENT_BETA", "0.5"))
                gate = 1.0 + beta * froth * float(direction)  # agree->>1, conflict-><1
                gate = min(max(gate, 0.0), 1.5)
                lam_eff *= gate
                sent_info = {"froth": round(froth, 3), "gate": round(gate, 3),
                             **{k: v for k, v in sent_info.items() if k == "sentiment"}}
            irf_sig = float(direction) * conviction           # signed, magnitude=conviction
            blend = (1.0 - lam_eff) * llm_sig + lam_eff * irf_sig
            denom = max(abs(llm_sig), _EPS)
            net_out = blend * envelope / denom
            info = {"irf": "leakfree", "direction": direction, "n_pre_t": n,
                    "conviction": round(conviction, 3), "lambda_eff": round(lam_eff, 3),
                    "llm_sig": round(llm_sig, 3), "irf_sig": round(irf_sig, 3),
                    "net_in": round(net_signal, 3), "net_out": round(net_out, 3),
                    **sent_info}
            return net_out, info

        # ── v1 (original, leaky strength) — kept for ablation/back-compat ──
        strength_wf = (self.recalibrate_strength(event_type, industry_code,
                                                 event_dt, base_strength)
                       if walk_forward else base_strength)
        irf_sig = float(direction) * abs(strength_wf) if direction != 0 else strength_wf
        if direction == 0:
            irf_sig = strength_wf
        blend = (1.0 - self.blend_lambda) * llm_sig + self.blend_lambda * irf_sig
        denom = max(abs(llm_sig), _EPS)
        net_out = blend * envelope / denom if abs(llm_sig) > _EPS else blend * envelope
        info = {
            "irf": "applied", "direction": direction,
            "base_strength": round(base_strength, 3),
            "strength_wf": round(strength_wf, 3),
            "llm_sig": round(llm_sig, 3), "irf_sig": round(irf_sig, 3),
            "blend": round(blend, 3), "net_in": round(net_signal, 3),
            "net_out": round(net_out, 3), "lambda": self.blend_lambda,
        }
        return net_out, info
