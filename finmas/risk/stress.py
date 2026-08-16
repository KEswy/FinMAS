"""Lightweight stress-test utilities for v5 portfolio weights."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from ..data.feature_store import FeatureStore


def _industry_returns() -> pd.DataFrame:
    industry = pd.read_csv("data/raw/industry_daily.csv")
    market = pd.read_csv("data/raw/hs300.csv")
    industry["日期"] = pd.to_datetime(industry["日期"])
    market["日期"] = pd.to_datetime(market["日期"])
    industry["industry_code"] = industry["industry_code"].astype(str)
    industry = industry.sort_values(["industry_code", "日期"])
    industry["ret"] = industry.groupby("industry_code")["收盘"].pct_change()
    wide = industry.pivot_table(
        index="日期", columns="industry_code", values="ret", aggfunc="last"
    ).sort_index()
    return wide


def industry_betas(as_of_date: str, window: int = 60) -> Dict[str, float]:
    market = pd.read_csv("data/raw/hs300.csv")
    market["日期"] = pd.to_datetime(market["日期"])
    market["ret"] = market["收盘"].pct_change()
    market = market.dropna().set_index("日期")["ret"].sort_index()
    wide = _industry_returns()
    as_of = pd.Timestamp(as_of_date)
    prior = wide.index[wide.index < as_of]
    if len(prior) < 30:
        return {str(c): 1.0 for c in wide.columns}
    sample = prior[-window:]
    out: Dict[str, float] = {}
    for col in wide.columns:
        y = wide.loc[sample, col].to_numpy(dtype=float)
        x = market.loc[sample].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 20:
            out[str(col)] = 1.0
            continue
        beta = float(np.cov(x[mask], y[mask], ddof=1)[0, 1] /
                     np.var(x[mask], ddof=1))
        out[str(col)] = beta if np.isfinite(beta) else 1.0
    return out


def run_synthetic_stress(
    weights: Dict[str, float],
    market_shock: float,
    as_of_date: str = "2025-12-31",
    vol_multiplier: float = 1.0,
) -> Dict[str, float]:
    betas = industry_betas(as_of_date)
    pnl = 0.0
    worst = []
    for industry, weight in weights.items():
        shock = float(betas.get(str(industry), 1.0)) * market_shock * vol_multiplier
        contribution = float(weight) * shock
        pnl += contribution
        worst.append(
            {
                "industry_code": str(industry),
                "weight": float(weight),
                "beta": float(betas.get(str(industry), 1.0)),
                "shock": shock,
                "pnl_contribution": contribution,
            }
        )
    worst.sort(key=lambda x: x["pnl_contribution"])
    return {
        "market_shock": market_shock,
        "portfolio_pnl": float(pnl),
        "worst_industries": worst[:10],
    }


def run_historical_stress(
    weights: Dict[str, float],
    start_date: str,
    end_date: str,
) -> Dict[str, float]:
    wide = _industry_returns()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    mask = (wide.index >= start) & (wide.index <= end)
    pnl = 0.0
    worst = []
    for industry, weight in weights.items():
        col = str(industry)
        if col not in wide.columns:
            continue
        rets = wide.loc[mask, col].fillna(0.0).to_numpy(dtype=float)
        cum = float(np.prod(1.0 + rets) - 1.0)
        contribution = float(weight) * cum
        pnl += contribution
        worst.append(
            {
                "industry_code": col,
                "weight": float(weight),
                "return": cum,
                "pnl_contribution": contribution,
            }
        )
    worst.sort(key=lambda x: x["pnl_contribution"])
    return {
        "start_date": start_date,
        "end_date": end_date,
        "portfolio_pnl": float(pnl),
        "worst_industries": worst[:10],
    }

