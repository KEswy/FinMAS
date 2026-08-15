"""
Fundamentals-scraping skill (v4 Phase 2, part 1: price-derived proxy)
====================================================================

Gathers, per (industry, event) and using ONLY data strictly before the event
date t, the observable state of the financial system's layers as reflected in
prices — a firewall-clean fundamentals proxy. Part 2 (scraping real
fundamentals: earnings / macro prints, each pub_date-gated) is a separate,
larger effort tracked independently.

Why price-derived first: build_feature_frame already computes ret / vol_chg /
volatility / mkt_ret / price_ma5 / price_ma20, and slice_history_before_event
enforces strict `< event_dt`. So this proxy is temporally firewalled by
construction (the price/windowing path was already the one clean channel in v3)
and needs no new data.

The probe emits a structured, layered read:
  - market layer:   relative strength vs CSI300, volatility regime
  - sector layer:   momentum (20d), valuation position (MA20 deviation)
  - flow layer:     volume-change trend
plus a scalar ``pre_state`` in [-1,1] (bullish↔bearish pre-event drift) and a
``confidence`` in [0,1] (how pronounced / one-sided the pre-event state is).

These feed the TransmissionMap cell (path + confidence), NOT the direction —
Phase 2 keeps direction LLM-driven to isolate its effect (per the plan).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import math

import numpy as np
import pandas as pd

from windowing import build_feature_frame, slice_history_before_event


@dataclass
class FundamentalsProbe:
    industry_code: str
    event_date: str
    momentum_20d: float = 0.0        # cumulative 20d return before t
    rel_strength: float = 0.0        # industry 20d ret − market 20d ret
    vol_regime: float = 0.0          # recent vol / historical vol − 1 (>0 = elevated)
    valuation_pos: float = 0.0       # MA20 deviation (last value; <0 = price above MA20)
    volume_trend: float = 0.0        # mean recent vol_chg
    pre_state: float = 0.0           # signed [-1,1] pre-event drift/strength
    confidence: float = 0.0          # [0,1] how pronounced the state is
    n_hist: int = 0                  # <t history length (cold-start flag)
    layers: List[str] = field(default_factory=list)   # human-readable layer notes

    def to_dict(self) -> dict:
        return {
            "momentum_20d": round(self.momentum_20d, 5),
            "rel_strength": round(self.rel_strength, 5),
            "vol_regime": round(self.vol_regime, 4),
            "valuation_pos": round(self.valuation_pos, 5),
            "volume_trend": round(self.volume_trend, 5),
            "pre_state": round(self.pre_state, 4),
            "confidence": round(self.confidence, 4),
            "n_hist": self.n_hist,
            "layers": self.layers,
        }


class FundamentalsSkill:
    """Price-derived, firewall-clean pre-event fundamentals probe."""

    def __init__(self, lookback: int = 20,
                 val_csv: str = "data/raw/industry_fundamentals.csv"):
        self.lookback = lookback
        self._frames: dict = {}   # industry_code -> feature_df cache
        self.val_csv = val_csv
        self._val_df = None       # lazy-loaded industry valuation table

    def _frame(self, industry_code: str) -> Optional[pd.DataFrame]:
        ic = str(industry_code)
        if ic not in self._frames:
            try:
                self._frames[ic] = build_feature_frame(ic)
            except Exception:
                self._frames[ic] = None
        return self._frames[ic]

    def probe(self, industry_code: str, event_date: str) -> FundamentalsProbe:
        """Compute the pre-event fundamentals proxy using only <t data."""
        pr = FundamentalsProbe(industry_code=str(industry_code), event_date=str(event_date)[:10])
        fdf = self._frame(industry_code)
        if fdf is None:
            return pr
        try:
            hist = slice_history_before_event(fdf, event_date)   # strict < event_dt
        except Exception:
            return pr
        pr.n_hist = len(hist)
        if pr.n_hist < self.lookback:
            return pr   # cold start: not enough <t history → neutral probe

        recent = hist.iloc[-self.lookback:]
        ret = recent["ret"].to_numpy()
        mkt = recent["mkt_ret"].to_numpy()

        pr.momentum_20d = float(np.nansum(ret))
        pr.rel_strength = float(np.nansum(ret) - np.nansum(mkt))
        # vol regime: recent vol vs full <t history vol
        recent_vol = float(np.nanstd(ret) + 1e-8)
        hist_vol = float(np.nanstd(hist["ret"].to_numpy()) + 1e-8)
        pr.vol_regime = float(recent_vol / hist_vol - 1.0)
        pr.valuation_pos = float(recent["price_ma20"].iloc[-1])   # (MA20/price − 1)
        pr.volume_trend = float(np.nanmean(recent["vol_chg"].to_numpy()))

        # pre_state: signed drift strength, squashed to [-1,1]. Combines
        # own-momentum and relative strength (both pre-event, firewall-clean).
        raw = 0.6 * pr.momentum_20d + 0.4 * pr.rel_strength
        pr.pre_state = float(math.tanh(raw / 0.05))   # 0.05 ≈ typical 20d move scale

        # confidence: pronounced when drift is large AND vol regime not chaotic.
        drift_mag = min(abs(pr.pre_state), 1.0)
        calm = 1.0 / (1.0 + max(pr.vol_regime, 0.0))   # elevated vol → less confident
        pr.confidence = float(round(drift_mag * calm, 4))

        # layered human-readable notes (market → sector → flow)
        pr.layers = [
            f"market: rel_strength={pr.rel_strength:+.3f}, "
            f"vol_regime={'elevated' if pr.vol_regime > 0.2 else 'normal'}",
            f"sector: momentum_20d={pr.momentum_20d:+.3f}, "
            f"valuation={'below-MA20' if pr.valuation_pos < 0 else 'above-MA20'}",
            f"flow: volume_trend={pr.volume_trend:+.3f}",
        ]
        return pr

    def apply_to_map(self, tm, industry_code: str, event_date: str):
        """Attach the probe to the TransmissionMap cell: writes the fundamentals
        path notes and nudges the cell's confidence. Does NOT change direction
        (Phase 2 isolates the fundamentals contribution to confidence/evidence).
        Returns the FundamentalsProbe for logging."""
        pr = self.probe(industry_code, event_date)
        cell = tm.cells.get(str(industry_code)) if tm and getattr(tm, "cells", None) else None
        if cell is not None:
            cell.path = list(cell.path) + pr.layers
            # blend fundamentals confidence into the cell (average, bounded)
            cell.confidence = float(min(1.0, 0.5 * cell.confidence + 0.5 * pr.confidence)) \
                if cell.confidence else pr.confidence
        return pr

    # ── part3 (A): REAL industry valuation probe (PE/PB/dividend, <t only) ──
    def _val_table(self):
        if self._val_df is None:
            try:
                d = pd.read_csv(self.val_csv, dtype={"industry_code": str})
                d["_d"] = pd.to_datetime(d["date"])
                self._val_df = d
            except Exception:
                self._val_df = False   # sentinel: unavailable
        return self._val_df if self._val_df is not False else None

    def probe_valuation(self, industry_code: str, event_date: str) -> "ValuationProbe":
        """Real-fundamentals probe: PE/PB history percentile + valuation momentum,
        using ONLY rows strictly before event_date (firewall-clean).

        valuation_signal in [-1,1]: extreme-cheap (low PE percentile) → +
        (margin of safety); extreme-rich (high percentile) → − (priced-in /
        vulnerable). A real-data candidate to correct the LLM's positive bias
        on nominally-bullish events where the sector is already expensive."""
        vp = ValuationProbe(industry_code=str(industry_code),
                            event_date=str(event_date)[:10])
        tbl = self._val_table()
        if tbl is None:
            return vp
        t = pd.to_datetime(event_date)
        sub = tbl[(tbl["industry_code"] == str(industry_code)) & (tbl["_d"] < t)]
        sub = sub.dropna(subset=["pe_ttm"])
        vp.n_hist = len(sub)
        if vp.n_hist < 60:            # <~3 months of <t history → no percentile
            return vp
        pe = sub["pe_ttm"].to_numpy(dtype=float)
        pe_now = float(pe[-1])
        # historical percentile of current PE within <t history (0=cheap,1=rich)
        vp.pe_percentile = float((pe < pe_now).mean())
        if "pb" in sub.columns and sub["pb"].notna().any():
            pb = sub["pb"].dropna().to_numpy(dtype=float)
            vp.pb_percentile = float((pb < pb[-1]).mean())
        if "dividend_yield" in sub.columns and sub["dividend_yield"].notna().any():
            vp.dividend_yield = float(sub["dividend_yield"].dropna().iloc[-1])
        # 20d PE momentum (rising valuation = getting more expensive)
        if len(pe) >= 21 and pe[-21] != 0:
            vp.pe_momentum_20d = float(pe_now / pe[-21] - 1.0)

        # valuation_signal: cheap→+, rich→−, centered at median. Scaled so the
        # extremes (0/1 percentile) map to ±1.
        vp.valuation_signal = float(-(vp.pe_percentile - 0.5) * 2.0)
        # confidence: how extreme (far from median) the valuation is
        vp.confidence = float(round(abs(vp.pe_percentile - 0.5) * 2.0, 4))
        vp.note = (f"valuation: PE={pe_now:.1f} (pctile={vp.pe_percentile:.0%}), "
                   f"PB pctile={vp.pb_percentile:.0%}, "
                   f"div={vp.dividend_yield:.2f}%, PE_20d_mom={vp.pe_momentum_20d:+.1%}")
        return vp

    def apply_valuation_to_map(self, tm, industry_code: str, event_date: str):
        """Attach the REAL-valuation probe to the cell (path + confidence).
        Does NOT change direction (isolate contribution). Returns the probe."""
        vp = self.probe_valuation(industry_code, event_date)
        cell = tm.cells.get(str(industry_code)) if tm and getattr(tm, "cells", None) else None
        if cell is not None and vp.note:
            cell.path = list(cell.path) + [vp.note]
            cell.confidence = float(min(1.0, 0.5 * cell.confidence + 0.5 * vp.confidence)) \
                if cell.confidence else vp.confidence
        return vp


@dataclass
class ValuationProbe:
    industry_code: str
    event_date: str
    pe_percentile: float = 0.5       # <t historical percentile of current PE (0=cheap,1=rich)
    pb_percentile: float = 0.5
    dividend_yield: float = 0.0
    pe_momentum_20d: float = 0.0
    valuation_signal: float = 0.0    # [-1,1]: cheap→+, rich→−
    confidence: float = 0.0          # [0,1]: how far from median valuation
    n_hist: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "pe_percentile": round(self.pe_percentile, 4),
            "pb_percentile": round(self.pb_percentile, 4),
            "dividend_yield": round(self.dividend_yield, 4),
            "pe_momentum_20d": round(self.pe_momentum_20d, 5),
            "valuation_signal": round(self.valuation_signal, 4),
            "confidence": round(self.confidence, 4),
            "n_hist": self.n_hist,
            "note": self.note,
        }
