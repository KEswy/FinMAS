"""Event, industry, and portfolio risk aggregation for FinMAS v5."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..data.feature_store import FeatureStore
from .engine import backtest_var


def _key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["event_date"].astype(str).str[:10]
        + "|"
        + frame["industry_code"].astype(str)
    )


def enrich_v5_risk(
    result_df: pd.DataFrame,
    horizons: Sequence[int] = (1, 5, 20),
    scales: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    """Attach predicted v5 VaR and realized CAR for each horizon."""
    scales = scales or {1: 0.4, 5: -0.4, 20: -0.4}
    store = FeatureStore()
    car = pd.read_csv("data/processed/car_results_expanded.csv")
    car = car.drop_duplicates(["event_date", "industry_code", "window"])
    car["event_date"] = car["event_date"].astype(str).str[:10]
    car["industry_code"] = car["industry_code"].astype(str)

    df = result_df.copy()
    df["event_date"] = df["event_date"].astype(str).str[:10]
    df["industry_code"] = df["industry_code"].astype(str)

    for horizon in horizons:
        car_h = car[car["window"] == int(horizon)][
            ["event_date", "industry_code", "CAR"]
        ].rename(columns={"CAR": f"realized_car_T{horizon}"})
        df = df.merge(car_h, on=["event_date", "industry_code"], how="left")

    vars = []
    for _, row in df.iterrows():
        matrix = store.pre_event_matrix(
            row["industry_code"], row["event_date"], lookback=120
        )
        sigma = (
            float(np.nanstd(matrix["ret"]))
            if matrix is not None and not matrix.empty
            else 0.01
        )
        confidence = float(row.get("confidence", 0.0))
        out = {}
        for horizon in horizons:
            scale_value = float(scales.get(int(horizon), -0.4))
            scale = 1.0 + scale_value * max(0.0, 1.0 - confidence)
            var = -1.645 * sigma * np.sqrt(int(horizon)) * scale
            out[f"pred_var_T{horizon}"] = var
            out[f"pred_es_T{horizon}"] = var * 1.15
        vars.append(out)
    risk_cols = pd.DataFrame(vars)
    return pd.concat([df.reset_index(drop=True), risk_cols.reset_index(drop=True)], axis=1)


def aggregate_portfolio_risk(
    enriched: pd.DataFrame,
    horizons: Sequence[int] = (1, 5, 20),
    weights: Optional[Sequence[float]] = None,
) -> dict:
    n = len(enriched)
    w = np.asarray(weights if weights is not None else [1.0 / n] * n, dtype=float)
    if w.sum() > 0:
        w = w / w.sum()
    else:
        w = np.ones(n) / n

    out: Dict[str, object] = {
        "n": n,
        "weights": w.tolist(),
        "horizons": {},
        "industry": {},
    }
    for horizon in horizons:
        realized = enriched[f"realized_car_T{horizon}"].to_numpy(dtype=float)
        pred_var = enriched[f"pred_var_T{horizon}"].to_numpy(dtype=float)
        pred_es = enriched[f"pred_es_T{horizon}"].to_numpy(dtype=float)
        mask = np.isfinite(realized) & np.isfinite(pred_var)
        realized_m = realized[mask]
        pred_var_m = pred_var[mask]
        pred_es_m = pred_es[mask]
        w_m = w[mask]
        w_m = w_m / w_m.sum() if w_m.sum() > 0 else w_m
        portfolio_realized = float(np.sum(realized_m * w_m)) if len(w_m) else np.nan
        portfolio_var = float(np.sum(pred_var_m * w_m)) if len(w_m) else np.nan
        portfolio_es = float(np.sum(pred_es_m * w_m)) if len(w_m) else np.nan
        bt = backtest_var(realized_m, pred_var_m, alpha=0.05)
        out["horizons"][str(horizon)] = {
            "realized_car": round(portfolio_realized, 8),
            "pred_var": round(portfolio_var, 8),
            "pred_es": round(portfolio_es, 8),
            "backtest": bt,
        }

        industry_rows = []
        for code, grp in enriched.groupby("industry_code"):
            sub_realized = grp[f"realized_car_T{horizon}"].to_numpy(dtype=float)
            sub_var = grp[f"pred_var_T{horizon}"].to_numpy(dtype=float)
            sub_w = w[grp.index]
            mask_i = np.isfinite(sub_realized) & np.isfinite(sub_var)
            if mask_i.any():
                sub_w = sub_w[mask_i] / max(sub_w[mask_i].sum(), 1e-12)
                industry_rows.append(
                    {
                        "industry_code": str(code),
                        "n": int(mask_i.sum()),
                        "realized_car": float(np.sum(sub_realized[mask_i] * sub_w)),
                        "pred_var": float(np.sum(sub_var[mask_i] * sub_w)),
                    }
                )
        industry_rows.sort(key=lambda x: x["realized_car"])
        out["industry"][str(horizon)] = industry_rows
    return out
