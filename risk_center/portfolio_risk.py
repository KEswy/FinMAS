"""Event, industry, and portfolio risk aggregation."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from evaluation.risk_metrics import backtest_var

from .data_loader import RiskCenterData, RealizedPath
from .risk_models import estimate_risk


def _pred_col(row: dict, base: str, horizon: int) -> Optional[float]:
    value = row.get(f"{base}_T{horizon}")
    if value is not None and pd.notna(value):
        return float(value)
    if horizon == 5:
        legacy = row.get(base)
        if legacy is not None and pd.notna(legacy):
            return float(legacy)
    return None


def _interval_col(row: dict, horizon: int) -> Optional[Tuple[float, float]]:
    lo = _pred_col(row, "risk_lo", horizon)
    hi = _pred_col(row, "risk_hi", horizon)
    if lo is None or hi is None:
        return None
    return float(lo), float(hi)


def resolve_row_weights(rows: List[dict], weights: Optional[dict] = None) -> List[float]:
    n = len(rows)
    if n == 0:
        return []
    if not weights:
        return [1.0 / n] * n

    raw = [0.0] * n
    if "by_industry" in weights:
        mapping = {str(k): float(v) for k, v in weights.get("by_industry", {}).items()}
        for i, row in enumerate(rows):
            raw[i] = mapping.get(str(row["industry_code"]), 0.0)
    elif "by_event" in weights:
        mapping = {}
        for item in weights.get("by_event", []):
            key = (
                str(item.get("event_date", ""))[:10],
                str(item.get("industry_code", "")),
            )
            mapping[key] = float(item.get("weight", 0.0))
        for i, row in enumerate(rows):
            key = (str(row.get("event_date", ""))[:10], str(row.get("industry_code", "")))
            raw[i] = mapping.get(key, 0.0)
    else:
        raise ValueError("weights must contain 'by_industry' or 'by_event'")

    arr = np.asarray(raw, dtype=float)
    total = float(arr.sum())
    if total <= 0 or not np.isfinite(total):
        return [1.0 / n] * n
    return (arr / total).tolist()


def industry_weights_from_rows(rows: List[dict], row_weights: List[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row, weight in zip(rows, row_weights):
        code = str(row["industry_code"])
        out[code] = out.get(code, 0.0) + float(weight)
    return out


def enrich_rows(
    data: RiskCenterData,
    horizons: Sequence[int],
) -> List[dict]:
    rows: List[dict] = []
    df = data.experiment_rows()
    for _, raw in df.iterrows():
        event_date = str(raw["event_date"])[:10]
        industry_code = str(raw["industry_code"])
        row = {
            "event_date": event_date,
            "event_name": str(raw.get("event_name", "")),
            "industry_code": industry_code,
            "event_type": raw.get("event_type", ""),
            "real_dir": raw.get("real_dir", ""),
            "final_dir": raw.get("final_dir", ""),
            "dir_correct": bool(raw.get("dir_correct", False)),
            "dispersion": float(raw.get("dispersion", 0.0)),
        }
        for horizon in horizons:
            realized: Optional[RealizedPath] = data.realized_path(
                industry_code, event_date, int(horizon)
            )
            if realized is None:
                row[f"realized_car_T{horizon}"] = np.nan
                row[f"realized_mdd_T{horizon}"] = np.nan
                row[f"realized_downside_T{horizon}"] = np.nan
                row[f"industry_return_T{horizon}"] = np.nan
                row[f"market_return_T{horizon}"] = np.nan
            else:
                row[f"realized_car_T{horizon}"] = realized.active_car
                row[f"realized_mdd_T{horizon}"] = realized.active_mdd
                row[f"realized_downside_T{horizon}"] = realized.active_downside
                row[f"industry_return_T{horizon}"] = realized.industry_cum_return
                row[f"market_return_T{horizon}"] = realized.market_cum_return

            row[f"pred_var_T{horizon}"] = _pred_col(raw.to_dict(), "event_VaR", int(horizon))
            row[f"pred_mdd_T{horizon}"] = _pred_col(raw.to_dict(), "expected_MDD", int(horizon))
            row[f"pred_tail_T{horizon}"] = _pred_col(raw.to_dict(), "tail_prob", int(horizon))
            row[f"pred_interval_T{horizon}"] = _interval_col(raw.to_dict(), int(horizon))
        rows.append(row)
    return rows


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    vals = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(vals)
    if not np.any(mask):
        return float("nan")
    return float(np.sum(vals[mask] * w[mask]) / max(np.sum(w[mask]), 1e-12))


def _portfolio_daily_returns(
    data: RiskCenterData,
    industry_weights: Dict[str, float],
) -> pd.Series:
    out = pd.Series(0.0, index=data.dates, dtype=float)
    for industry, weight in industry_weights.items():
        if industry in data.industry_returns.columns and weight != 0:
            out = out + weight * data.industry_returns[industry].fillna(0.0)
    return out


def aggregate_risk(
    rows: List[dict],
    data: RiskCenterData,
    weights: Optional[dict] = None,
    horizons: Sequence[int] = (1, 5, 20),
    alphas: Sequence[float] = (0.05, 0.01),
    methods: Sequence[str] = ("historical", "bootstrap", "parametric", "ewma", "evt", "model_interval"),
    limits: Optional[Dict[str, float]] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    horizons = tuple(int(h) for h in horizons)
    alphas = tuple(float(a) for a in alphas)
    methods = tuple(str(m) for m in methods)
    row_weights = resolve_row_weights(rows, weights)
    industry_weights = industry_weights_from_rows(rows, row_weights)
    portfolio_returns = _portfolio_daily_returns(data, industry_weights)

    portfolio_risk: Dict[str, dict] = {}
    calibration: Dict[str, dict] = {}
    industry_risk: Dict[str, list] = {}
    event_risk: Dict[str, list] = {}

    for horizon in horizons:
        realized_car = [row.get(f"realized_car_T{horizon}", np.nan) for row in rows]
        realized_mdd = [row.get(f"realized_mdd_T{horizon}", np.nan) for row in rows]
        realized_downside = [row.get(f"realized_downside_T{horizon}", np.nan) for row in rows]
        pred_var = [row.get(f"pred_var_T{horizon}", np.nan) for row in rows]
        pred_mdd = [row.get(f"pred_mdd_T{horizon}", np.nan) for row in rows]

        portfolio_realized_car = _weighted_mean(realized_car, row_weights)
        portfolio_realized_mdd = _weighted_mean(realized_mdd, row_weights)
        portfolio_realized_es = _weighted_mean(realized_downside, row_weights)
        portfolio_pred_var = _weighted_mean(pred_var, row_weights)
        portfolio_pred_mdd = _weighted_mean(pred_mdd, row_weights)

        method_by_alpha: Dict[str, dict] = {}
        for alpha in alphas:
            method_by_alpha[str(alpha)] = {}
            for method in methods:
                if method == "model_interval":
                    method_by_alpha[str(alpha)][str(method)] = {
                        "var": portfolio_pred_var,
                        "es": portfolio_pred_var,
                    }
                else:
                    estimate = estimate_risk(
                        portfolio_returns.to_numpy(dtype=float),
                        float(alpha),
                        str(method),
                        n_bootstrap,
                        seed,
                    )
                    method_by_alpha[str(alpha)][str(method)] = {
                        "var": float(estimate["var"]) * np.sqrt(float(horizon)),
                        "es": float(estimate["es"]) * np.sqrt(float(horizon)),
                    }

        portfolio_risk[str(horizon)] = {
            "realized_car": round(portfolio_realized_car, 8),
            "realized_mdd": round(portfolio_realized_mdd, 8),
            "realized_es": round(portfolio_realized_es, 8),
            "predicted_var": round(portfolio_pred_var, 8) if np.isfinite(portfolio_pred_var) else None,
            "predicted_mdd": round(portfolio_pred_mdd, 8) if np.isfinite(portfolio_pred_mdd) else None,
            "hhi": round(float(np.sum(np.asarray(row_weights) ** 2)), 8),
            "effective_n": round(float(1.0 / max(np.sum(np.asarray(row_weights) ** 2), 1e-12)), 2),
            "methods": method_by_alpha,
        }

        r = np.asarray(realized_car, dtype=float)
        v = np.asarray(pred_var, dtype=float)
        mask = np.isfinite(r) & np.isfinite(v)
        if int(mask.sum()) > 0:
            bt = backtest_var(
                r[mask],
                v[mask],
                alpha=0.05,
                realized_mdd=np.asarray(realized_mdd, dtype=float)[mask],
                pred_mdd=np.asarray(pred_mdd, dtype=float)[mask],
            ).to_dict()
        else:
            bt = None
        calibration[str(horizon)] = bt

        frame = pd.DataFrame(
            {
                "industry_code": [str(row["industry_code"]) for row in rows],
                "weight": row_weights,
                "realized_car": realized_car,
                "realized_mdd": realized_mdd,
                "realized_downside": realized_downside,
                "pred_var": pred_var,
            }
        )
        ind_rows = []
        for industry, grp in frame.groupby("industry_code"):
            ind_rows.append(
                {
                    "industry_code": industry,
                    "weight": float(grp["weight"].sum()),
                    "realized_car": _weighted_mean(grp["realized_car"], grp["weight"]),
                    "realized_mdd": _weighted_mean(grp["realized_mdd"], grp["weight"]),
                    "realized_es": _weighted_mean(grp["realized_downside"], grp["weight"]),
                    "predicted_var": _weighted_mean(grp["pred_var"], grp["weight"]),
                    "n_events": int(len(grp)),
                }
            )
        ind_rows.sort(key=lambda x: x["realized_car"])
        industry_risk[str(horizon)] = ind_rows

        event_risk[str(horizon)] = sorted(
            rows,
            key=lambda row: float(row.get(f"realized_car_T{horizon}", np.inf)),
        )

    limit_report = evaluate_limits(industry_weights, calibration, portfolio_risk, limits)
    return {
        "portfolio_risk": portfolio_risk,
        "calibration": calibration,
        "industry_risk": industry_risk,
        "event_risk": event_risk,
        "limits": limit_report,
        "weights": {
            "by_industry": {k: round(v, 8) for k, v in sorted(industry_weights.items())}
        },
    }


def evaluate_limits(
    industry_weights: Dict[str, float],
    calibration: Dict[str, Optional[dict]],
    portfolio_risk: Dict[str, dict],
    limits: Optional[Dict[str, float]] = None,
) -> dict:
    limits = limits or {}
    warnings = []
    single_limit = float(limits.get("single_industry_weight", 0.20))
    hhi_limit = float(limits.get("hhi_warning", 0.10))
    breach_multiple = float(limits.get("breach_warning_multiple", 2.0))
    var_budget = float(limits.get("portfolio_var_budget", 0.10))

    for industry, weight in industry_weights.items():
        if weight > single_limit:
            warnings.append(
                {
                    "type": "single_industry_weight",
                    "industry_code": industry,
                    "value": round(weight, 6),
                    "limit": single_limit,
                }
            )

    hhi = float(np.sum(np.asarray(list(industry_weights.values())) ** 2))
    if hhi > hhi_limit:
        warnings.append(
            {
                "type": "hhi",
                "value": round(hhi, 6),
                "limit": hhi_limit,
            }
        )

    for horizon, backtest in calibration.items():
        if not backtest:
            continue
        expected = float(backtest.get("expected_rate", 0.05))
        breach = float(backtest.get("breach_rate", 0.0))
        if expected > 0 and breach > expected * breach_multiple:
            warnings.append(
                {
                    "type": "var_breach",
                    "horizon": int(horizon),
                    "breach_rate": breach,
                    "expected_rate": expected,
                    "multiple": breach / expected,
                }
            )

    for horizon, risk in portfolio_risk.items():
        var_05 = None
        methods = risk.get("methods", {}).get("0.05", {})
        for method, value in methods.items():
            var_05 = value.get("var")
            break
        if var_05 is not None and abs(float(var_05)) > var_budget:
            warnings.append(
                {
                    "type": "portfolio_var_budget",
                    "horizon": int(horizon),
                    "var": round(float(var_05), 6),
                    "limit": var_budget,
                }
            )

    return {
        "hhi": round(hhi, 8),
        "warnings": warnings,
        "breach_warning": len([w for w in warnings if w["type"] == "var_breach"]) > 0,
        "concentration_warning": hhi > hhi_limit,
    }
