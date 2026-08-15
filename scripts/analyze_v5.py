"""Analyze a FinMAS v5 walk-forward result file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    parser.add_argument("--result-csv", required=True)
    parser.add_argument(
        "--legacy-csv",
        default="data/processed/all_experiment_results_deepseek-v4-flash_wf_deepseek_flash_321_gpu_20260815_204839.csv",
    )
    parser.add_argument("--nboot", type=int, default=2000)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    df = join_event_type(pd.read_csv(args.result_csv))
    legacy = join_event_type(pd.read_csv(args.legacy_csv))
    report = {
        "summary": summary(df),
        "selective_curve": selective_curve(df),
        "paired_bootstrap_full": paired_bootstrap(legacy, df, n_boot=args.nboot),
        "paired_bootstrap_committed": committed_paired_bootstrap(legacy, df, n_boot=args.nboot),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

