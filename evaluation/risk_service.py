"""
Offline risk-report service for walk-forward experiment results.

The service consumes an existing experiment CSV, joins it with the daily
industry/market price files, and produces an auditable risk report with:

  * realized T+1 / T+5 / T+20 CAR, path MDD, and downside;
  * pooled VaR backtests when predicted risk columns are available;
  * event-level, industry-level, and equal-weight portfolio views;
  * top risk contributors and concentration.

It is intentionally read-only with respect to the existing experiment CSV and
only writes new report artifacts under ``data/risk_reports/``.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from evaluation.risk_metrics import backtest_var


_DEFAULT_HORIZONS = (1, 5, 20)


@dataclass
class RiskReportConfig:
    horizons: Tuple[int, ...] = _DEFAULT_HORIZONS
    alpha: float = 0.05
    output_dir: str = "data/risk_reports"
    weights_path: Optional[str] = None
    industry_daily_csv: str = "data/raw/industry_daily.csv"
    market_daily_csv: str = "data/raw/hs300.csv"
    output_csv: bool = True
    n_bootstrap: int = 1000
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


class _MarketData:
    """Preloads aligned industry and CSI300 daily returns once per report."""

    def __init__(self, industry_csv: str, market_csv: str):
        industry = pd.read_csv(industry_csv)
        market = pd.read_csv(market_csv)

        industry = industry.rename(
            columns={"日期": "date", "收盘": "close", "industry_code": "industry_code"}
        )
        market = market.rename(columns={"日期": "date", "收盘": "close"})

        industry["date"] = pd.to_datetime(industry["date"])
        market["date"] = pd.to_datetime(market["date"])
        industry["industry_code"] = industry["industry_code"].astype(str)

        industry_prices = (
            industry.pivot_table(
                index="date",
                columns="industry_code",
                values="close",
                aggfunc="last",
            )
            .sort_index()
        )
        market_prices = (
            market[["date", "close"]]
            .drop_duplicates("date")
            .set_index("date")["close"]
            .sort_index()
        )

        industry_returns = industry_prices.pct_change(fill_method=None)
        market_returns = market_prices.pct_change(fill_method=None)

        common_dates = sorted(set(industry_returns.index) & set(market_returns.index))
        self.dates = pd.DatetimeIndex(common_dates)
        self.industry_returns = industry_returns.loc[self.dates]
        self.market_returns = market_returns.loc[self.dates]

    def realized_path_metrics(
        self, industry_code: str, event_date: str, horizon: int
    ) -> Optional[dict]:
        """Compute active-return CAR and path drawdown for one event/industry.

        ``CAR`` in this project is industry cumulative return minus market
        cumulative return. We therefore build the active-return path as
        ``(1 + industry_return) / (1 + market_return) - 1``, whose compounded
        value equals the project's CAR definition.
        """
        if horizon <= 0:
            return None
        industry_code = str(industry_code)
        if industry_code not in self.industry_returns.columns:
            return None

        event_dt = pd.Timestamp(str(event_date)[:10])
        pos = int(self.dates.searchsorted(event_dt, side="left"))
        end_pos = pos + horizon
        if pos >= len(self.dates) or end_pos >= len(self.dates):
            return None

        ind_rets = self.industry_returns.iloc[pos : end_pos + 1][industry_code]
        mkt_rets = self.market_returns.iloc[pos : end_pos + 1]
        ind_rets = ind_rets.fillna(0.0).to_numpy(dtype=float)
        mkt_rets = mkt_rets.fillna(0.0).to_numpy(dtype=float)

        active_rets = (1.0 + ind_rets) / (1.0 + mkt_rets) - 1.0
        active_rets = np.nan_to_num(active_rets, nan=0.0, posinf=0.0, neginf=0.0)

        active_path = np.cumprod(1.0 + active_rets) - 1.0
        nav = 1.0 + active_path
        roll_max = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
        drawdown = (roll_max - nav) / np.maximum(roll_max, 1e-12)
        realized_mdd = float(np.clip(np.max(drawdown), 0.0, 1.0))
        realized_car = float(active_path[-1])

        return {
            "t0": str(self.dates[pos].date()),
            "realized_car": realized_car,
            "realized_mdd": realized_mdd,
            "realized_downside": max(-realized_car, 0.0),
            "n_returns": int(len(active_rets)),
        }


def load_weights(path: str) -> dict:
    """Load an optional external weight JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize(weights: Sequence[float]) -> List[float]:
    arr = np.asarray(weights, dtype=float)
    total = float(arr.sum())
    if total <= 0 or not np.isfinite(total):
        return [1.0 / max(len(arr), 1)] * len(arr)
    return (arr / total).tolist()


def _resolve_weights(rows: List[dict], weights: Optional[dict]) -> List[float]:
    """Resolve row weights to a normalized list.

    Defaults to equal-weight rows. ``by_industry`` maps an industry to a raw
    weight; ``by_event`` matches exact ``event_date`` + ``industry_code``.
    """
    n = len(rows)
    if n == 0:
        return []
    if not weights:
        return [1.0 / n] * n

    raw = [0.0] * n
    if "by_industry" in weights:
        industry_weights = {str(k): float(v) for k, v in weights["by_industry"].items()}
        for i, row in enumerate(rows):
            raw[i] = industry_weights.get(str(row["industry_code"]), 0.0)
    elif "by_event" in weights:
        event_map = {}
        for item in weights["by_event"]:
            key = (str(item.get("event_date", ""))[:10], str(item.get("industry_code", "")))
            event_map[key] = float(item.get("weight", 0.0))
        for i, row in enumerate(rows):
            key = (str(row.get("event_date", ""))[:10], str(row.get("industry_code", "")))
            raw[i] = event_map.get(key, 0.0)
    else:
        raise ValueError("weights must contain 'by_industry' or 'by_event'")

    return _normalize(raw)


def _pred_col(row: dict, base: str, horizon: int) -> Optional[float]:
    """Read either a horizon-specific column or the legacy T+5 column."""
    horizon_col = f"{base}_T{horizon}"
    if horizon_col in row and pd.notna(row.get(horizon_col)):
        return float(row[horizon_col])
    if horizon == 5 and base in row and pd.notna(row.get(base)):
        return float(row[base])
    return None


def _interval_col(row: dict, horizon: int) -> Optional[Tuple[float, float]]:
    lo = _pred_col(row, "risk_lo", horizon)
    hi = _pred_col(row, "risk_hi", horizon)
    if lo is None or hi is None:
        return None
    return float(lo), float(hi)


def _bootstrap_breach_ci(
    realized: Sequence[float],
    var: Sequence[float],
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float]:
    r = np.asarray(realized, dtype=float)
    v = np.asarray(var, dtype=float)
    n = len(r)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(max(int(n_bootstrap), 0)):
        idx = rng.integers(0, n, n)
        rates.append(float(np.mean(r[idx] < v[idx])))
    if not rates:
        return float(np.mean(r < v)), float(np.mean(r < v))
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    vals = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(vals)
    if not np.any(mask):
        return float("nan")
    return float(np.sum(vals[mask] * w[mask]) / max(np.sum(w[mask]), 1e-12))


def _safe_backtest_dict(
    realized: Sequence[float],
    var: Sequence[float],
    realized_mdd: Optional[Sequence[float]],
    pred_mdd: Optional[Sequence[float]],
    alpha: float,
    n_bootstrap: int,
    seed: int,
) -> Optional[dict]:
    r = np.asarray(realized, dtype=float)
    v = np.asarray(var, dtype=float)
    mask = np.isfinite(r) & np.isfinite(v)
    if int(mask.sum()) == 0:
        return None
    r = r[mask]
    v = v[mask]
    bt = backtest_var(
        r,
        v,
        alpha=alpha,
        realized_mdd=realized_mdd[mask] if realized_mdd is not None else None,
        pred_mdd=pred_mdd[mask] if pred_mdd is not None else None,
    )
    out = bt.to_dict()
    ci_lo, ci_hi = _bootstrap_breach_ci(r, v, n_bootstrap, seed)
    out["breach_rate_ci_low"] = ci_lo
    out["breach_rate_ci_high"] = ci_hi
    return out


def aggregate_risk(
    rows: List[dict],
    weights: Optional[dict] = None,
    horizons: Sequence[int] = _DEFAULT_HORIZONS,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Aggregate already-enriched row dictionaries into a risk report."""
    horizons = tuple(int(h) for h in horizons)
    row_weights = _resolve_weights(rows, weights)
    n = len(rows)
    if n == 0:
        return {
            "portfolio_risk": {},
            "calibration": {},
            "industry_risk": {},
            "attribution": {},
            "dispersion": {},
        }

    portfolio_risk = {}
    calibration = {}
    industry_risk = {}
    attribution = {}
    dispersion = {}

    for h in horizons:
        realized_car = [row.get(f"realized_car_T{h}", np.nan) for row in rows]
        realized_mdd = [row.get(f"realized_mdd_T{h}", np.nan) for row in rows]
        realized_downside = [row.get(f"realized_downside_T{h}", np.nan) for row in rows]
        pred_var = [row.get(f"pred_var_T{h}", np.nan) for row in rows]
        pred_mdd = [row.get(f"pred_mdd_T{h}", np.nan) for row in rows]

        realized_car_arr = np.asarray(realized_car, dtype=float)
        realized_mdd_arr = np.asarray(realized_mdd, dtype=float)
        realized_downside_arr = np.asarray(realized_downside, dtype=float)
        pred_var_arr = np.asarray(pred_var, dtype=float)
        pred_mdd_arr = np.asarray(pred_mdd, dtype=float)

        w = np.asarray(row_weights, dtype=float)
        portfolio_realized_car = _weighted_mean(realized_car_arr, w)
        portfolio_realized_mdd = _weighted_mean(realized_mdd_arr, w)
        portfolio_realized_es = _weighted_mean(realized_downside_arr, w)
        portfolio_pred_var = _weighted_mean(pred_var_arr, w)
        portfolio_pred_mdd = _weighted_mean(pred_mdd_arr, w)

        portfolio_risk[str(h)] = {
            "realized_car": round(portfolio_realized_car, 6),
            "realized_mdd": round(portfolio_realized_mdd, 6),
            "realized_es": round(portfolio_realized_es, 6),
            "predicted_var": round(portfolio_pred_var, 6)
            if np.isfinite(portfolio_pred_var)
            else None,
            "predicted_mdd": round(portfolio_pred_mdd, 6)
            if np.isfinite(portfolio_pred_mdd)
            else None,
            "hhi": round(float(np.sum(w**2)), 6),
            "effective_n": round(float(1.0 / max(np.sum(w**2), 1e-12)), 2),
        }

        calibration[str(h)] = _safe_backtest_dict(
            realized_car_arr,
            pred_var_arr,
            realized_mdd_arr,
            pred_mdd_arr,
            alpha,
            n_bootstrap,
            seed,
        )

        frame = pd.DataFrame(
            {
                "industry_code": [str(r["industry_code"]) for r in rows],
                "weight": row_weights,
                "realized_car": realized_car_arr,
                "realized_mdd": realized_mdd_arr,
                "realized_downside": realized_downside_arr,
                "pred_var": pred_var_arr,
            }
        )
        industry_rows = []
        for industry, grp in frame.groupby("industry_code"):
            industry_rows.append(
                {
                    "industry_code": industry,
                    "weight": float(grp["weight"].sum()),
                    "realized_car": _weighted_mean(grp["realized_car"], grp["weight"]),
                    "realized_mdd": _weighted_mean(grp["realized_mdd"], grp["weight"]),
                    "realized_es": _weighted_mean(
                        grp["realized_downside"], grp["weight"]
                    ),
                    "predicted_var": _weighted_mean(grp["pred_var"], grp["weight"]),
                    "n_events": int(len(grp)),
                }
            )
        industry_rows.sort(key=lambda x: x["realized_car"])
        industry_risk[str(h)] = [
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
            for r in industry_rows
        ]

        top_events = sorted(
            rows,
            key=lambda r: float(r.get(f"realized_car_T{h}", np.inf)),
        )[:10]
        attribution[str(h)] = {
            "top_loss_events": [
                {
                    "event_date": r.get("event_date"),
                    "event_name": r.get("event_name"),
                    "industry_code": r.get("industry_code"),
                    "realized_car": round(
                        float(r.get(f"realized_car_T{h}", np.nan)), 6
                    ),
                    "realized_mdd": round(
                        float(r.get(f"realized_mdd_T{h}", np.nan)), 6
                    ),
                }
                for r in top_events
            ],
            "top_loss_industries": industry_risk[str(h)][:10],
        }

        disp = pd.to_numeric(
            pd.Series([r.get("dispersion", np.nan) for r in rows]),
            errors="coerce",
        ).dropna()
        dispersion[str(h)] = {
            "mean": round(float(disp.mean()), 6) if not disp.empty else None,
            "p50": round(float(disp.median()), 6) if not disp.empty else None,
            "p90": round(float(disp.quantile(0.9)), 6) if not disp.empty else None,
            "max": round(float(disp.max()), 6) if not disp.empty else None,
        }

    return {
        "portfolio_risk": portfolio_risk,
        "calibration": calibration,
        "industry_risk": industry_risk,
        "attribution": attribution,
        "dispersion": dispersion,
    }


def _enrich_rows(
    df: pd.DataFrame, market_data: _MarketData, horizons: Sequence[int]
) -> List[dict]:
    rows: List[dict] = []
    for _, raw in df.iterrows():
        row = {
            "event_date": str(raw.get("event_date", ""))[:10],
            "event_name": str(raw.get("event_name", ""))[:80],
            "industry_code": str(raw.get("industry_code", "")),
            "dispersion": float(raw.get("dispersion", np.nan))
            if pd.notna(raw.get("dispersion"))
            else np.nan,
        }
        for h in horizons:
            realized = market_data.realized_path_metrics(
                row["industry_code"], row["event_date"], int(h)
            )
            row[f"realized_car_T{h}"] = (
                realized["realized_car"] if realized else np.nan
            )
            row[f"realized_mdd_T{h}"] = (
                realized["realized_mdd"] if realized else np.nan
            )
            row[f"realized_downside_T{h}"] = (
                realized["realized_downside"] if realized else np.nan
            )
            row[f"realized_t0_T{h}"] = realized["t0"] if realized else None
            row[f"pred_var_T{h}"] = _pred_col(raw, "event_VaR", int(h))
            row[f"pred_mdd_T{h}"] = _pred_col(raw, "expected_MDD", int(h))
            row[f"pred_tail_T{h}"] = _pred_col(raw, "tail_prob", int(h))
            interval = _interval_col(raw, int(h))
            row[f"pred_interval_T{h}"] = interval
        rows.append(row)
    return rows


def build_risk_report(csv_path: str, config: Optional[RiskReportConfig] = None) -> dict:
    """Build a risk report from an existing experiment CSV.

    Returns the report dict and writes ``<stem>_risk_report.json`` (and
    optionally ``<stem>_risk_report.csv``) under ``config.output_dir``.
    """
    config = config or RiskReportConfig()
    csv_path = str(csv_path)
    df = pd.read_csv(csv_path)
    required = {"event_date", "industry_code"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    market_data = _MarketData(config.industry_daily_csv, config.market_daily_csv)
    weights = load_weights(config.weights_path) if config.weights_path else None
    rows = _enrich_rows(df, market_data, config.horizons)
    report = aggregate_risk(
        rows,
        weights=weights,
        horizons=config.horizons,
        alpha=config.alpha,
        n_bootstrap=config.n_bootstrap,
        seed=config.seed,
    )

    stem = Path(csv_path).stem
    report["meta"] = {
        "source_csv": str(csv_path),
        "stem": stem,
        "generated_at": pd.Timestamp.now().isoformat(),
        "alpha": config.alpha,
        "horizons": list(config.horizons),
        "weighting": "equal_weight" if weights is None else weights,
        "n_events": len(rows),
        "n_industries": len({r["industry_code"] for r in rows}),
        "config": config.to_dict(),
    }
    report["event_risk"] = {
        str(h): [
            {
                "event_date": r["event_date"],
                "event_name": r["event_name"],
                "industry_code": r["industry_code"],
                "realized_car": r.get(f"realized_car_T{h}"),
                "realized_mdd": r.get(f"realized_mdd_T{h}"),
                "predicted_var": r.get(f"pred_var_T{h}"),
                "predicted_mdd": r.get(f"pred_mdd_T{h}"),
                "tail_prob": r.get(f"pred_tail_T{h}"),
                "risk_interval": r.get(f"pred_interval_T{h}"),
            }
            for r in sorted(
                rows,
                key=lambda r: float(r.get(f"realized_car_T{h}", np.inf)),
            )
        ]
        for h in config.horizons
    }

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}_risk_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    if config.output_csv:
        flat_rows = []
        for r in rows:
            flat = dict(r)
            for h in config.horizons:
                interval = flat.pop(f"pred_interval_T{h}", None)
                if interval:
                    flat[f"risk_lo_T{h}"] = interval[0]
                    flat[f"risk_hi_T{h}"] = interval[1]
            flat_rows.append(flat)
        pd.DataFrame(flat_rows).to_csv(
            output_dir / f"{stem}_risk_report.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return report
