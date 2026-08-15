"""
5b robustness check (reviewer): does the per-class result survive dropping the
market_event / geopolitical cells whose event_name embeds the realized move?

We recompute overall / positive / negative CAR-sign accuracy for Ours (3-seed
majority vote of final_dir) and LLM-only (single run), on three subsets:
  (A) full 168,
  (B) excluding market_event,
  (C) excluding market_event AND geopolitical.
Pure CPU from logged CSVs -- no re-run.

CLI: python scripts/robustness_exclude_market_event.py
"""
from __future__ import annotations
import os, sys, io, glob
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR); os.chdir(ROOT)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

BASE = [
    "data/processed/all_experiment_results_qwen2.5_wf_base_20260717_163837.csv",
    "data/processed/all_experiment_results_qwen2.5_wf_base_seed123_20260718_105105.csv",
    "data/processed/all_experiment_results_qwen2.5_wf_base_seed2024_20260718_212215.csv",
]


def keyed(df):
    df = df.copy()
    df["industry_code"] = df["industry_code"].astype(str)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.strftime("%Y-%m-%d")
    return df.set_index(["event_date", "industry_code"])


def acc(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ov = (y_true == y_pred).mean()
    pos, neg = y_true == 1, y_true == -1
    pa = (y_pred[pos] == 1).mean() if pos.any() else np.nan
    na = (y_pred[neg] == -1).mean() if neg.any() else np.nan
    return ov, pa, na, len(y_true), int(pos.sum()), int(neg.sum())


def main():
    car = pd.read_csv("data/processed/car_results.csv")
    car = car[car["window"] == 5].copy()
    car = keyed(car)
    et = car["event_type"]; y = np.where(car["CAR"] >= 0, 1, -1)
    spine = pd.DataFrame({"event_type": et.values,
                          "y": y}, index=car.index)

    # Ours: 3-seed majority vote of final_dir
    votes = []
    for f in BASE:
        b = keyed(pd.read_csv(f))
        v = b["final_dir"].map({"+": 1, "-": -1}).reindex(spine.index)
        votes.append(v)
    V = pd.concat(votes, axis=1)
    spine["ours"] = np.sign(V.sum(axis=1)).replace(0, 1).astype(int)

    # LLM-only single run
    llm_f = sorted(glob.glob("data/processed/baseline_llm_only_*.csv"))[-1]
    llm = keyed(pd.read_csv(llm_f))
    spine["llm"] = llm["pred_dir"].map({"+": 1, "-": -1}).reindex(spine.index)

    subsets = {
        "(A) full":                        spine,
        "(B) excl market_event":           spine[spine.event_type != "market_event"],
        "(C) excl market_event+geopol":    spine[~spine.event_type.isin(["market_event", "geopolitical"])],
    }
    print("=" * 78)
    print("5b robustness: per-class accuracy dropping outcome-in-name subsets")
    print(f"LLM-only file: {llm_f}")
    print("=" * 78)
    for name, s in subsets.items():
        s = s.dropna(subset=["ours", "llm"])
        oo = acc(s.y, s.ours); ll = acc(s.y, s.llm)
        print(f"\n{name}  (n={oo[3]}, +{oo[4]}/-{oo[5]})")
        print(f"  {'':13} overall    pos      neg")
        print(f"  Ours (maj)  {oo[0]*100:6.2f}%  {oo[1]*100:6.2f}%  {oo[2]*100:6.2f}%")
        print(f"  LLM-only    {ll[0]*100:6.2f}%  {ll[1]*100:6.2f}%  {ll[2]*100:6.2f}%")
        print(f"  gap(O-L)    {(oo[0]-ll[0])*100:+6.2f}   {(oo[1]-ll[1])*100:+6.2f}   {(oo[2]-ll[2])*100:+6.2f}")
    # event_type composition
    print("\nevent_type counts (window=5):")
    print(spine.event_type.value_counts().to_string())


if __name__ == "__main__":
    main()
