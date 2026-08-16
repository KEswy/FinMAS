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
) -> pd.DataFrame:
    thresholds = list(thresholds or np.linspace(0.0, 0.5, 21))
    rows = []
    for threshold in thresholds:
        df = ensemble_results(a, b, weights=weights, threshold=threshold)
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
