"""Run portfolio/industry risk aggregation on a v5 result CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from finmas.risk.portfolio import aggregate_portfolio_risk, enrich_v5_risk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-csv",
        default="data/processed/all_experiment_results_finmas_v5_oof_ensemble_abstain_market.csv",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.result_csv)
    committed = df[~df["abstain"]].copy() if "abstain" in df.columns else df.copy()
    enriched = enrich_v5_risk(committed)
    report = aggregate_portfolio_risk(enriched)
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

