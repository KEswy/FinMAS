"""Historical and synthetic stress tests for an industry portfolio."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .data_loader import RiskCenterData


def _round_float(value: float, ndigits: int = 8) -> float:
    if value is None or not np.isfinite(value):
        return float("nan")
    return round(float(value), ndigits)


def run_stress_tests(
    data: RiskCenterData,
    industry_weights: Dict[str, float],
    scenarios: Optional[Dict] = None,
    as_of_date: Optional[str] = None,
    beta_window: Optional[int] = None,
) -> Dict[str, List[dict]]:
    scenarios = scenarios or {}
    betas = data.industry_betas(as_of_date or "2025-12-31", window=beta_window)
    results: Dict[str, List[dict]] = {"historical": [], "synthetic": []}

    for item in scenarios.get("historical", []):
        start = str(item.get("start_date", ""))
        end = str(item.get("end_date", ""))
        if not start or not end:
            continue
        scenario_returns = data.historical_scenario_returns(start, end)
        pnl = 0.0
        worst: List[dict] = []
        for industry, weight in industry_weights.items():
            industry_ret = float(scenario_returns.get(industry, 0.0))
            contrib = weight * industry_ret
            pnl += contrib
            worst.append(
                {
                    "industry_code": industry,
                    "weight": round(weight, 6),
                    "return": round(industry_ret, 6),
                    "pnl_contribution": round(contrib, 6),
                }
            )
        worst.sort(key=lambda x: x["pnl_contribution"])
        results["historical"].append(
            {
                "name": item.get("name", f"{start}_{end}"),
                "start_date": start,
                "end_date": end,
                "market_return": round(float(scenario_returns.get("HS300", np.nan)), 6),
                "portfolio_pnl": _round_float(pnl),
                "worst_industries": worst[:10],
            }
        )

    for item in scenarios.get("synthetic", []):
        market_shock = float(item.get("market_shock", 0.0))
        vol_multiplier = float(item.get("vol_multiplier", 1.0))
        pnl = 0.0
        worst: List[dict] = []
        for industry, weight in industry_weights.items():
            beta = float(betas.get(industry, 1.0))
            industry_shock = beta * market_shock * vol_multiplier
            contrib = weight * industry_shock
            pnl += contrib
            worst.append(
                {
                    "industry_code": industry,
                    "weight": round(weight, 6),
                    "beta": round(beta, 4),
                    "shock": round(industry_shock, 6),
                    "pnl_contribution": round(contrib, 6),
                }
            )
        worst.sort(key=lambda x: x["pnl_contribution"])
        results["synthetic"].append(
            {
                "name": item.get("name", "synthetic"),
                "market_shock": market_shock,
                "vol_multiplier": vol_multiplier,
                "portfolio_pnl": _round_float(pnl),
                "worst_industries": worst[:10],
            }
        )

    return results
