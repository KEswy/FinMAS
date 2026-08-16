"""Run synthetic and historical stress tests for final v5 committed weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from finmas.risk.stress import run_historical_stress, run_synthetic_stress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-csv",
        default="data/processed/all_experiment_results_finmas_v5_final_ensemble.csv",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.result_csv)
    committed = df[~df["abstain"]].copy() if "abstain" in df.columns else df.copy()
    counts = committed["industry_code"].value_counts()
    weights = {str(code): float(n / counts.sum()) for code, n in counts.items()}

    report = {
        "weights": weights,
        "synthetic": [
            run_synthetic_stress(weights, -0.05, vol_multiplier=1.0),
            run_synthetic_stress(weights, -0.10, vol_multiplier=1.0),
            run_synthetic_stress(weights, -0.03, vol_multiplier=2.0),
        ],
        "historical": [
            run_historical_stress(weights, "2020-02-03", "2020-03-23"),
            run_historical_stress(weights, "2024-09-24", "2024-10-08"),
            run_historical_stress(weights, "2025-04-02", "2025-04-09"),
        ],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

