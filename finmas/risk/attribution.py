"""Return attribution for v5 event/industry results."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .stress import industry_betas


def run_return_attribution(
    result_df: pd.DataFrame,
    horizons: Sequence[int] = (1, 5, 20),
) -> Dict[str, list]:
    car = pd.read_csv("data/processed/car_results_expanded.csv")
    car = car.drop_duplicates(["event_date", "industry_code", "window"])
    car["event_date"] = car["event_date"].astype(str).str[:10]
    car["industry_code"] = car["industry_code"].astype(str)

    rows = result_df.copy()
    rows["event_date"] = rows["event_date"].astype(str).str[:10]
    rows["industry_code"] = rows["industry_code"].astype(str)

    output: Dict[str, list] = {}
    for horizon in horizons:
        car_h = car[car["window"] == int(horizon)].copy()
        merged = rows.merge(
            car_h,
            on=["event_date", "industry_code"],
            how="inner",
            suffixes=("", "_car"),
        )
        attribution_rows = []
        for _, row in merged.iterrows():
            beta = float(
                industry_betas(row["event_date"]).get(
                    str(row["industry_code"]), 1.0
                )
            )
            market_cum = float(row["cum_mkt_ret"])
            industry_cum = float(row["cum_ind_ret"])
            active_car = float(row["CAR"])
            market_component = (beta - 1.0) * market_cum
            industry_relative = industry_cum - beta * market_cum
            event_alpha = active_car - market_component - industry_relative
            attribution_rows.append(
                {
                    "event_date": row["event_date"],
                    "event_type": row.get("event_type", ""),
                    "industry_code": str(row["industry_code"]),
                    "active_car": active_car,
                    "industry_return": industry_cum,
                    "market_return": market_cum,
                    "beta": beta,
                    "market_component": market_component,
                    "industry_relative": industry_relative,
                    "event_alpha": event_alpha,
                }
            )
        output[str(horizon)] = attribution_rows
    return output

