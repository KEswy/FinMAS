"""Backtest a simple v5 event-level VaR against legacy event_VaR_T5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from finmas.data.feature_store import FeatureStore
from finmas.risk.engine import backtest_var, kupiec_pof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-csv",
        default="data/processed/all_experiment_results_finmas_v5_oof_ensemble_abstain_market.csv",
    )
    parser.add_argument(
        "--legacy-csv",
        default="data/processed/all_experiment_results_deepseek-v4-flash_wf_deepseek_flash_321_gpu_20260815_204839.csv",
    )
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--scale", type=float, default=-0.4)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    store = FeatureStore()
    v5 = pd.read_csv(args.result_csv)
    legacy = pd.read_csv(args.legacy_csv)
    legacy["event_date"] = legacy["event_date"].astype(str).str[:10]
    legacy["industry_code"] = legacy["industry_code"].astype(str)
    v5["event_date"] = v5["event_date"].astype(str).str[:10]
    v5["industry_code"] = v5["industry_code"].astype(str)
    merged = v5.merge(
        legacy[["event_date", "industry_code", "real_CAR", "event_VaR_T5"]],
        on=["event_date", "industry_code"],
        how="inner",
        suffixes=("", "_legacy"),
    )

    rows = []
    for _, row in merged.iterrows():
        matrix = store.pre_event_matrix(
            row["industry_code"], row["event_date"], lookback=120
        )
        sigma = float(np.nanstd(matrix["ret"])) if matrix is not None and not matrix.empty else 0.01
        confidence = float(row.get("confidence", 0.0))
        scale = 1.0 + args.scale * max(0.0, 1.0 - confidence)
        var5 = -1.645 * sigma * np.sqrt(args.horizon) * scale
        rows.append(
            {
                "event_date": row["event_date"],
                "industry_code": row["industry_code"],
                "real_CAR": float(row["real_CAR"]),
                "v5_var": var5,
                "legacy_var": float(row["event_VaR_T5"]),
                "v5_breach": bool(float(row["real_CAR"]) < var5),
                "legacy_breach": bool(
                    float(row["real_CAR"]) < float(row["event_VaR_T5"])
                ),
            }
        )
    df = pd.DataFrame(rows)

    v5_bt = backtest_var(df["real_CAR"], df["v5_var"], alpha=0.05)
    legacy_bt = backtest_var(df["real_CAR"], df["legacy_var"], alpha=0.05)
    report = {
        "n": int(len(df)),
        "horizon": args.horizon,
        "scale": args.scale,
        "v5": v5_bt,
        "legacy": legacy_bt,
        "v5_breach_rate": float(df["v5_breach"].mean()),
        "legacy_breach_rate": float(df["legacy_breach"].mean()),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
