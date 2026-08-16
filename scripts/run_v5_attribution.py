"""Run native v5 return attribution for final ensemble results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from finmas.risk.attribution import run_return_attribution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-csv",
        default="data/processed/all_experiment_results_finmas_v5_final_ensemble.csv",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.result_csv)
    report = run_return_attribution(df)
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(text[:2000])
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

