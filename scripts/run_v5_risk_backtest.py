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
    parser.add_argument("--horizons", default="1,5,20")
    parser.add_argument("--scale", type=float, default=-0.4)
    parser.add_argument("--scale-1", type=float, default=None)
    parser.add_argument("--scale-5", type=float, default=None)
    parser.add_argument("--scale-20", type=float, default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    store = FeatureStore()
    v5 = pd.read_csv(args.result_csv)
    legacy = pd.read_csv(args.legacy_csv)
    car = pd.read_csv("data/processed/car_results_expanded.csv")
    car = car.drop_duplicates(["event_date", "industry_code", "window"])
    car["event_date"] = car["event_date"].astype(str).str[:10]
    car["industry_code"] = car["industry_code"].astype(str)
    legacy["event_date"] = legacy["event_date"].astype(str).str[:10]
    legacy["industry_code"] = legacy["industry_code"].astype(str)
    v5["event_date"] = v5["event_date"].astype(str).str[:10]
    v5["industry_code"] = v5["industry_code"].astype(str)
    report = {"scale": args.scale, "horizons": {}}
    for horizon_text in args.horizons.split(","):
        horizon = int(horizon_text.strip())
        scale_value = {
            1: args.scale_1 if args.scale_1 is not None else args.scale,
            5: args.scale_5 if args.scale_5 is not None else args.scale,
            20: args.scale_20 if args.scale_20 is not None else args.scale,
        }.get(horizon, args.scale)
        car_h = car[car["window"] == horizon][
            ["event_date", "industry_code", "CAR"]
        ].rename(columns={"CAR": "real_CAR"})
        legacy_col = f"event_VaR_T{horizon}"
        if legacy_col not in legacy.columns:
            continue
        merged = v5.merge(
            legacy[["event_date", "industry_code", legacy_col]],
            on=["event_date", "industry_code"],
            how="inner",
        ).merge(car_h, on=["event_date", "industry_code"], how="inner")

        rows = []
        for _, row in merged.iterrows():
            matrix = store.pre_event_matrix(
                row["industry_code"], row["event_date"], lookback=120
            )
            sigma = (
                float(np.nanstd(matrix["ret"]))
                if matrix is not None and not matrix.empty
                else 0.01
            )
            confidence = float(row.get("confidence", 0.0))
            scale = 1.0 + scale_value * max(0.0, 1.0 - confidence)
            var = -1.645 * sigma * np.sqrt(horizon) * scale
            rows.append(
                {
                    "event_date": row["event_date"],
                    "industry_code": row["industry_code"],
                    "real_CAR": float(row["real_CAR"]),
                    "v5_var": var,
                    "legacy_var": float(row[legacy_col]),
                    "v5_breach": bool(float(row["real_CAR"]) < var),
                    "legacy_breach": bool(
                        float(row["real_CAR"]) < float(row[legacy_col])
                    ),
                }
            )
        df = pd.DataFrame(rows)
        report["horizons"][str(horizon)] = {
            "n": int(len(df)),
            "v5": backtest_var(df["real_CAR"], df["v5_var"], alpha=0.05),
            "legacy": backtest_var(df["real_CAR"], df["legacy_var"], alpha=0.05),
            "v5_breach_rate": float(df["v5_breach"].mean()),
            "legacy_breach_rate": float(df["legacy_breach"].mean()),
        }
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
