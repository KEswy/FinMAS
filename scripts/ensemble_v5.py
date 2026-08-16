"""Ensemble two FinMAS v5 probability result files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from finmas.ensemble import ensemble_results, threshold_scan
from finmas.reporting import (
    committed_paired_bootstrap,
    join_event_type,
    paired_bootstrap,
    selective_curve,
    summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--weight-a", type=float, default=0.5)
    parser.add_argument("--weight-b", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--legacy-csv",
        default="data/processed/all_experiment_results_deepseek-v4-flash_wf_deepseek_flash_321_gpu_20260815_204839.csv",
    )
    args = parser.parse_args()

    a = pd.read_csv(args.a)
    b = pd.read_csv(args.b)
    df = ensemble_results(
        a,
        b,
        weights=(args.weight_a, args.weight_b),
        threshold=args.threshold,
    )
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    legacy = join_event_type(pd.read_csv(args.legacy_csv))
    report = {
        "weights": [args.weight_a, args.weight_b],
        "threshold": args.threshold,
        "summary": summary(join_event_type(df)),
        "selective_curve": selective_curve(join_event_type(df)),
        "threshold_scan": threshold_scan(
            a,
            b,
            weights=(args.weight_a, args.weight_b),
        ).to_dict("records"),
        "paired_bootstrap_full": paired_bootstrap(legacy, df, n_boot=2000),
        "paired_bootstrap_committed": committed_paired_bootstrap(legacy, df, n_boot=2000),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

