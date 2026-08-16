"""Simple calibrated ensemble of two direction-probability result files."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def _key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["event_date"].astype(str).str[:10]
        + "|"
        + frame["industry_code"].astype(str)
    )


def ensemble_results(
    a: pd.DataFrame,
    b: pd.DataFrame,
    weights: Sequence[float] = (0.5, 0.5),
    threshold: float = 0.12,
    abstain_event_types: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Average two probability estimates on aligned event/industry rows."""
    wa, wb = float(weights[0]), float(weights[1])
    total = wa + wb
    wa, wb = wa / total, wb / total

    left = a.copy()
    right = b.copy()
    left["_key"] = _key(left)
    right["_key"] = _key(right)
    merged = left.merge(
        right[["_key", "prob_up", "abstain", "dir_correct", "real_dir"]],
        on="_key",
        how="inner",
        suffixes=("_a", "_b"),
    )
    merged["prob_up"] = wa * merged["prob_up_a"] + wb * merged["prob_up_b"]
    merged["confidence"] = (merged["prob_up"] - 0.5).abs() * 2.0
    merged["abstain"] = merged["confidence"] < threshold
    if abstain_event_types:
        event_type_col = (
            "event_type"
            if "event_type" in merged.columns
            else "event_type_a"
            if "event_type_a" in merged.columns
            else "event_type_b"
        )
        merged["abstain"] = merged["abstain"] | merged[event_type_col].isin(
            set(abstain_event_types)
        )
    merged["final_dir"] = np.where(merged["prob_up"] >= 0.5, "+", "-")
    real_dir = merged["real_dir_a"] if "real_dir_a" in merged else merged["real_dir_b"]
    merged["real_dir"] = real_dir
    merged["dir_correct"] = (merged["final_dir"] == merged["real_dir"]).astype(int)
    merged["committed_dir_correct"] = (
        (~merged["abstain"]) & (merged["final_dir"] == merged["real_dir"])
    ).astype(int)
    event_type_col = (
        "event_type"
        if "event_type" in merged.columns
        else "event_type_a"
        if "event_type_a" in merged.columns
        else "event_type_b"
    )
    cols = [
        "event_date",
        "industry_code",
        event_type_col,
        "real_dir",
        "final_dir",
        "prob_up",
        "confidence",
        "abstain",
        "dir_correct",
        "committed_dir_correct",
    ]
    out = merged[cols].rename(columns={event_type_col: "event_type"}).copy()
    return out


def threshold_scan(
    a: pd.DataFrame,
    b: pd.DataFrame,
    weights: Sequence[float] = (0.5, 0.5),
    thresholds: Optional[Sequence[float]] = None,
    abstain_event_types: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    thresholds = list(thresholds or np.linspace(0.0, 0.5, 21))
    rows = []
    for threshold in thresholds:
        df = ensemble_results(
            a,
            b,
            weights=weights,
            threshold=threshold,
            abstain_event_types=abstain_event_types,
        )
        committed = df[~df["abstain"]]
        rows.append(
            {
                "threshold": float(threshold),
                "full_accuracy": float(df["dir_correct"].mean()),
                "coverage": float(len(committed) / max(len(df), 1)),
                "committed_accuracy": float(committed["committed_dir_correct"].mean())
                if len(committed)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def walk_forward_ensemble(
    a: pd.DataFrame,
    b: pd.DataFrame,
    weights_grid: Sequence[float] = (0.2, 0.4, 0.5, 0.6, 0.8),
    threshold_grid: Sequence[float] = (0.05, 0.10, 0.12, 0.15, 0.20),
    abstain_event_types: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Choose ensemble weight and threshold strictly from past rows.

    This is a lightweight walk-forward stacking experiment: each event sees only
    information available before its date, so the resulting accuracy is a fair
    sample outside comparison.
    """
    left = a.copy()
    right = b.copy()
    left["_key"] = _key(left)
    right["_key"] = _key(right)
    merged = left.merge(
        right[["_key", "prob_up", "real_dir", "event_type"]],
        on="_key",
        how="inner",
        suffixes=("_a", "_b"),
    )
    merged = merged.sort_values("event_date").reset_index(drop=True)

    weights = [float(w) for w in weights_grid]
    thresholds = [float(t) for t in threshold_grid]
    past_prob_a: list[float] = []
    past_prob_b: list[float] = []
    past_real: list[str] = []
    past_types: list[str] = []
    rows = []

    for _, row in merged.iterrows():
        event_type = str(row.get("event_type") or row.get("event_type_a") or "")
        best_w = 0.6
        best_t = 0.12
        best_score = -1.0
        if len(past_real) >= 20:
            for w in weights:
                for t in thresholds:
                    probs = [
                        w * pa + (1.0 - w) * pb
                        for pa, pb in zip(past_prob_a, past_prob_b)
                    ]
                    conf = [abs(p - 0.5) * 2.0 for p in probs]
                    committed = [
                        i
                        for i, (cv, et) in enumerate(zip(conf, past_types))
                        if cv >= t
                        and (not abstain_event_types or et not in set(abstain_event_types))
                    ]
                    if len(committed) < 5:
                        continue
                    acc = float(np.mean([past_real[i] == ("+" if probs[i] >= 0.5 else "-")
                                         for i in committed]))
                    coverage = float(len(committed) / max(len(past_real), 1))
                    score = acc + 0.05 * coverage
                    if score > best_score:
                        best_score = score
                        best_w = w
                        best_t = t

        prob = best_w * float(row["prob_up_a"]) + (1.0 - best_w) * float(row["prob_up_b"])
        confidence = abs(prob - 0.5) * 2.0
        forced_abstain = bool(
            abstain_event_types and event_type in set(abstain_event_types)
        )
        abstain = forced_abstain or confidence < best_t
        direction = "+" if prob >= 0.5 else "-"
        real_dir = str(row["real_dir_a"])
        rows.append(
            {
                "event_date": row["event_date"],
                "industry_code": row["industry_code"],
                "event_type": event_type,
                "real_dir": real_dir,
                "final_dir": direction,
                "prob_up": prob,
                "confidence": confidence,
                "abstain": bool(abstain),
                "weight_a": best_w,
                "threshold": best_t,
                "dir_correct": bool(direction == real_dir),
                "committed_dir_correct": bool((not abstain) and direction == real_dir),
            }
        )
        past_prob_a.append(float(row["prob_up_a"]))
        past_prob_b.append(float(row["prob_up_b"]))
        past_real.append(real_dir)
        past_types.append(event_type)

    return pd.DataFrame(rows)
