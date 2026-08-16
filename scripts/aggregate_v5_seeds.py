"""Aggregate multi-seed probability files by averaging prob_up."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from finmas.reporting import (
    committed_paired_bootstrap,
    join_event_type,
    paired_bootstrap,
    selective_curve,
    summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, help="comma-separated CSV paths")
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--legacy-csv",
        default="data/processed/all_experiment_results_deepseek-v4-flash_wf_deepseek_flash_321_gpu_20260815_204839.csv",
    )
    args = parser.parse_args()

    frames = []
    for path in args.inputs.split(","):
        path = path.strip()
        if not path:
            continue
        df = pd.read_csv(path)
        df["_key"] = (
            df["event_date"].astype(str).str[:10]
            + "|"
            + df["industry_code"].astype(str)
        )
        frames.append(df[["_key", "prob_up", "real_dir", "event_type"]])

    if not frames:
        raise ValueError("no input files")

    base = frames[0].copy()
    for i, frame in enumerate(frames[1:], start=1):
        base = base.merge(
            frame,
            on="_key",
            how="inner",
            suffixes=("", f"_{i}"),
        )
    prob_cols = ["prob_up"] + [c for c in base.columns if c.startswith("prob_up_")]
    base["prob_up"] = base[prob_cols].mean(axis=1)
    base["confidence"] = (base["prob_up"] - 0.5).abs() * 2.0
    sign_cols = [f"prob_up" if i == 0 else f"prob_up_{i}" for i in range(len(frames))]
    signs = np.sign(base[sign_cols] - 0.5)
    disagreement = signs.nunique(axis=1) > 1
    base["abstain"] = (base["confidence"] < args.threshold) | disagreement
    base["final_dir"] = np.where(base["prob_up"] >= 0.5, "+", "-")
    base["real_dir"] = base["real_dir"]
    base["dir_correct"] = (base["final_dir"] == base["real_dir"]).astype(int)
    base["committed_dir_correct"] = (
        (~base["abstain"]) & (base["final_dir"] == base["real_dir"])
    ).astype(int)
    base["event_date"] = base["_key"].str.split("|").str[0]
    base["industry_code"] = base["_key"].str.split("|").str[1]
    out = base[["_key", "event_date", "industry_code", "event_type", "real_dir",
                "final_dir", "prob_up", "confidence", "abstain",
                "dir_correct", "committed_dir_correct"]].copy()
    out = out.drop(columns=["_key"])
    out.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    legacy = join_event_type(pd.read_csv(args.legacy_csv))
    report = {
        "summary": summary(join_event_type(out)),
        "selective_curve": selective_curve(join_event_type(out)),
        "paired_bootstrap_full": paired_bootstrap(legacy, out, n_boot=2000),
        "paired_bootstrap_committed": committed_paired_bootstrap(
            legacy, out, n_boot=2000
        ),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
