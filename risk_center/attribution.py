"""Return attribution and risk attribution for event/industry portfolios."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .data_loader import RiskCenterData


def run_return_attribution(
    data: RiskCenterData,
    rows: Sequence[dict],
    horizons: Sequence[int] = (1, 5, 20),
    beta_window: Optional[int] = None,
) -> Dict[str, list]:
    horizons = tuple(int(h) for h in horizons)
    output: Dict[str, list] = {}
    for horizon in horizons:
        attribution_rows: List[dict] = []
        for row in rows:
            industry = str(row["industry_code"])
            event_date = str(row["event_date"])[:10]
            realized = data.realized_path(industry, event_date, horizon)
            if realized is None:
                continue
            beta, fallback = data.pre_event_beta(
                industry, event_date, window=beta_window
            )
            market_component = (beta - 1.0) * realized.market_cum_return
            industry_relative = (
                realized.industry_cum_return - beta * realized.market_cum_return
            )
            event_alpha = (
                realized.active_car - market_component - industry_relative
            )
            attribution_rows.append(
                {
                    "event_date": event_date,
                    "event_name": row.get("event_name", ""),
                    "industry_code": industry,
                    "event_type": row.get("event_type", ""),
                    "active_car": round(realized.active_car, 8),
                    "industry_return": round(realized.industry_cum_return, 8),
                    "market_return": round(realized.market_cum_return, 8),
                    "beta": round(beta, 6),
                    "beta_fallback": bool(fallback),
                    "market_component": round(market_component, 8),
                    "industry_relative": round(industry_relative, 8),
                    "event_alpha": round(event_alpha, 8),
                }
            )
        output[str(horizon)] = attribution_rows
    return output


def aggregate_return_attribution(
    attribution: Dict[str, list],
    group_by: str = "event_type",
) -> Dict[str, list]:
    aggregated: Dict[str, list] = {}
    for horizon, rows in attribution.items():
        if not rows:
            aggregated[str(horizon)] = []
            continue
        frame = pd.DataFrame(rows)
        grouped = (
            frame.groupby(group_by)
            .agg(
                n=("active_car", "size"),
                active_car=("active_car", "mean"),
                market_component=("market_component", "mean"),
                industry_relative=("industry_relative", "mean"),
                event_alpha=("event_alpha", "mean"),
            )
            .reset_index()
        )
        aggregated[str(horizon)] = grouped.to_dict("records")
    return aggregated


def run_risk_attribution(
    data: RiskCenterData,
    industry_weights: Dict[str, float],
    alpha: float = 0.05,
    horizon: int = 5,
) -> Dict[str, list]:
    industries = list(data.industry_returns.columns)
    returns = data.industry_returns.fillna(0.0)
    cov = returns.cov().to_numpy(dtype=float)

    weights = np.zeros(len(industries), dtype=float)
    for i, industry in enumerate(industries):
        weights[i] = float(industry_weights.get(str(industry), 0.0))

    portfolio_var = float(weights @ cov @ weights)
    portfolio_vol = float(np.sqrt(max(portfolio_var, 0.0)))
    z = float(np.quantile(np.random.default_rng(0).normal(size=1000000), alpha))
    scale = np.sqrt(float(horizon))

    rows = []
    raw_component_var = []
    for i, industry in enumerate(industries):
        weight = float(weights[i])
        marginal = float((cov @ weights)[i] / portfolio_vol) if portfolio_vol > 0 else 0.0
        component = weight * marginal
        component_var = component * z * scale
        raw_component_var.append(component_var)
        rows.append(
            {
                "industry_code": str(industry),
                "weight": round(weight, 8),
                "marginal_var": round(marginal * z * scale, 8),
                "component_var": round(component_var, 8),
                "component_share": 0.0,
            }
        )
    total_risk = float(np.sum(np.abs(raw_component_var)))
    if total_risk > 0:
        for row, component_var in zip(rows, raw_component_var):
            row["component_share"] = round(float(abs(component_var) / total_risk), 8)
    rows.sort(key=lambda x: x["component_var"])
    return {
        "portfolio_vol_daily": round(portfolio_vol, 8),
        "portfolio_var": round(portfolio_vol * z * scale, 8),
        "components": rows,
    }
