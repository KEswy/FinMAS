"""
Ablation comparison (Tier M, n=168)
===================================

Lays out the full system (3-seed majority vote) against A1/A2/A3 ablations:
  A1 = no exogenous correction
  A2 = no KF branch (Informer only)
  A3 = no RAG

For each config: overall CAR-sign accuracy, per-event-type, per-direction.
Confirms the round-3 ablation directions hold at n=168 (RAG-off and
exogenous-off should both drop sharply).

All numbers from the result CSVs. No fabrication.

Usage:
  python scripts/analyze_ablation.py
"""
from __future__ import annotations
import os, sys, io

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
os.chdir(ROOT_DIR)

import pandas as pd
import numpy as np

OURS_FILES = {
    "seed42":   "data/processed/all_experiment_results_qwen2.5_v2_20260526_114901.csv",
    "seed123":  "data/processed/all_experiment_results_qwen2.5_v2_seed123_20260527_201449.csv",
    "seed2024": "data/processed/all_experiment_results_qwen2.5_v2_seed2024_20260528_185645.csv",
}
ABLATION_FILES = {
    "A1 (no exo)":  "data/processed/ablation_A1_tier_m_blind_results.csv",
    "A2 (no KF)":   "data/processed/ablation_A2_tier_m_blind_results.csv",
    "A3 (no RAG)":  "data/processed/ablation_A3_tier_m_blind_results.csv",
}


def _key(df):
    return (df["event_date"].astype(str).str[:10] + "|" +
            df["industry_code"].astype(str))


def event_type_map():
    """event_type lives in car_results.csv (window=5 rows)."""
    car = pd.read_csv("data/processed/car_results.csv")
    car = car[car["window"] == 5].copy()
    car["k"] = _key(car)
    return car.set_index("k")["event_type"].to_dict()


def load_full():
    """3-seed majority vote correctness + real_dir."""
    frames = []
    for name, path in OURS_FILES.items():
        d = pd.read_csv(path)
        d["k"] = _key(d)
        frames.append(d[["k", "real_dir", "dir_correct"]])
    alld = pd.concat(frames, ignore_index=True)
    g = alld.groupby("k").agg(
        real_dir=("real_dir", "first"),
        votes=("dir_correct", "sum"),
        n=("dir_correct", "count"),
    )
    g["correct"] = g["votes"] >= (g["n"] / 2.0)
    return g[["real_dir", "correct"]].reset_index()


def load_ablation(path):
    d = pd.read_csv(path)
    d["k"] = _key(d)
    d["correct"] = d["dir_correct"].astype(bool)
    return d[["k", "real_dir", "correct"]]


def summarize(df, et_map, label):
    df = df.copy()
    df["event_type"] = df["k"].map(et_map)
    n = len(df)
    overall = df["correct"].mean()
    pos = df[df["real_dir"] == "+"]["correct"]
    neg = df[df["real_dir"] == "-"]["correct"]
    print(f"\n{label}  (n={n})")
    print(f"  overall  {overall:.2%} ({df['correct'].sum()}/{n})")
    print(f"  positive {pos.mean():.2%} ({pos.sum()}/{len(pos)})  "
          f"negative {neg.mean():.2%} ({neg.sum()}/{len(neg)})")
    by = df.groupby("event_type")["correct"].agg(["mean", "count"])
    for et, r in by.iterrows():
        print(f"    {et:<22} {r['mean']:.2%}  (n={int(r['count'])})")
    return overall


def main() -> int:
    et_map = event_type_map()
    print("=" * 72)
    print("Ablation comparison (Tier M, n=168)")
    print("=" * 72)

    full = load_full()
    full_acc = summarize(full, et_map, "FULL (3-seed majority)")

    abl_acc = {}
    for label, path in ABLATION_FILES.items():
        if not os.path.exists(path):
            print(f"\n{label}: MISSING {path}")
            continue
        d = load_ablation(path)
        abl_acc[label] = summarize(d, et_map, label)

    print("\n" + "=" * 72)
    print("Drop vs FULL system (overall CAR-sign accuracy)")
    print("=" * 72)
    print(f"  FULL                {full_acc:.2%}")
    for label, acc in abl_acc.items():
        print(f"  {label:<18} {acc:.2%}   ({(acc-full_acc)*100:+.2f}pp)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
