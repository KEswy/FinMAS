"""Evaluation reporting and paired-bootstrap comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def _key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["event_date"].astype(str).str[:10]
        + "|"
        + frame["industry_code"].astype(str)
    )


def join_event_type(frame: pd.DataFrame, car_csv: str = "data/processed/car_results_expanded.csv") -> pd.DataFrame:
    car = pd.read_csv(car_csv)
    car = car[car["window"] == 5][["event_date", "industry_code", "event_type"]].drop_duplicates()
    car["event_date"] = car["event_date"].astype(str).str[:10]
    car["industry_code"] = car["industry_code"].astype(str)
    out = frame.copy()
    out["event_date"] = out["event_date"].astype(str).str[:10]
    out["industry_code"] = out["industry_code"].astype(str)
    if "event_type" in out.columns:
        out = out.drop(columns=["event_type"])
    return out.merge(car, on=["event_date", "industry_code"], how="left")


def summary(frame: pd.DataFrame) -> Dict[str, object]:
    df = frame.copy()
    full = {
        "n": int(len(df)),
        "accuracy": float(df["dir_correct"].mean()) if "dir_correct" in df else float("nan"),
    }
    committed = df[~df["abstain"]] if "abstain" in df else df
    selective = {
        "n": int(len(committed)),
        "coverage": float(len(committed) / max(len(df), 1)),
        "accuracy": float(committed["committed_dir_correct"].mean())
        if "committed_dir_correct" in committed else float("nan"),
    }
    out: Dict[str, object] = {
        "full": full,
        "selective": selective,
        "by_event_type": {},
        "by_industry": {},
    }
    if "event_type" in df.columns:
        def committed_stats(g: pd.DataFrame) -> pd.Series:
            c = g[~g["abstain"]]
            return pd.Series(
                {
                    "n": len(g),
                    "full_accuracy": g["dir_correct"].mean(),
                    "committed_n": len(c),
                    "committed_accuracy": c["committed_dir_correct"].mean()
                    if len(c)
                    else float("nan"),
                }
            )

        grouped = (
            df.groupby("event_type")
            .apply(committed_stats, include_groups=False)
            .sort_values("full_accuracy")
        )
        out["by_event_type"] = grouped.reset_index().to_dict("records")
    if "industry_code" in df.columns:
        grouped = (
            df.groupby("industry_code")
            .apply(committed_stats, include_groups=False)
            .sort_values("full_accuracy")
        )
        out["by_industry"] = grouped.reset_index().to_dict("records")
    return out


def selective_curve(frame: pd.DataFrame, n_bins: int = 10) -> List[Dict[str, float]]:
    df = frame.copy().sort_values("confidence", ascending=False)
    if "committed_dir_correct" not in df.columns:
        df["committed_dir_correct"] = df["dir_correct"]
    bins = np.linspace(0, len(df), n_bins + 1).astype(int)
    out = []
    for end in bins[1:]:
        if end <= 0:
            continue
        sub = df.iloc[:end]
        out.append(
            {
                "cumulative_rows": int(end),
                "coverage": float(end / max(len(df), 1)),
                "accuracy": float(sub["committed_dir_correct"].mean()),
                "min_confidence": float(sub["confidence"].min()),
            }
        )
    return out


def paired_bootstrap(
    legacy: pd.DataFrame,
    candidate: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, object]:
    """Compare candidate vs legacy on aligned rows with paired bootstrap."""
    legacy = join_event_type(legacy).copy()
    candidate = join_event_type(candidate).copy()
    legacy["_key"] = _key(legacy)
    candidate["_key"] = _key(candidate)
    merged = candidate.merge(
        legacy[["_key", "dir_correct"]].rename(columns={"dir_correct": "legacy_correct"}),
        on="_key",
        how="inner",
    )
    if merged.empty:
        return {"n": 0, "delta": float("nan"), "p_value": float("nan")}

    cand = merged["dir_correct"].astype(float).to_numpy()
    base = merged["legacy_correct"].astype(float).to_numpy()
    observed_delta = float(cand.mean() - base.mean())

    rng = np.random.default_rng(seed)
    deltas = []
    n = len(merged)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        deltas.append(float(cand[idx].mean() - base[idx].mean()))
    deltas = np.asarray(deltas)
    p_two_sided = float(np.mean(np.abs(deltas) >= abs(observed_delta)))
    return {
        "n": n,
        "candidate_accuracy": float(cand.mean()),
        "legacy_accuracy": float(base.mean()),
        "delta": observed_delta,
        "p_value": p_two_sided,
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
    }


def committed_paired_bootstrap(
    legacy: pd.DataFrame,
    candidate: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, object]:
    """Compare candidate committed subset to legacy on the same rows."""
    legacy = join_event_type(legacy).copy()
    candidate = join_event_type(candidate).copy()
    candidate = candidate[~candidate["abstain"]]
    legacy["_key"] = _key(legacy)
    candidate["_key"] = _key(candidate)
    merged = candidate.merge(
        legacy[["_key", "dir_correct"]].rename(columns={"dir_correct": "legacy_correct"}),
        on="_key",
        how="inner",
    )
    if merged.empty:
        return {"n": 0, "delta": float("nan"), "p_value": float("nan")}

    cand = merged["committed_dir_correct"].astype(float).to_numpy()
    base = merged["legacy_correct"].astype(float).to_numpy()
    observed_delta = float(cand.mean() - base.mean())
    rng = np.random.default_rng(seed)
    n = len(merged)
    deltas = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        deltas.append(float(cand[idx].mean() - base[idx].mean()))
    deltas = np.asarray(deltas)
    return {
        "n": n,
        "candidate_accuracy": float(cand.mean()),
        "legacy_accuracy": float(base.mean()),
        "delta": observed_delta,
        "p_value": float(np.mean(np.abs(deltas) >= abs(observed_delta))),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
    }
