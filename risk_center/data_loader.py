"""Data access layer for risk-center analytics.

This module reuses existing raw data and experiment CSV files. It does not run
or retrain any model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import RiskCenterConfig


@dataclass
class RealizedPath:
    active_car: float
    active_mdd: float
    active_downside: float
    industry_cum_return: float
    market_cum_return: float
    t0: str
    n_returns: int

    def to_dict(self) -> dict:
        return {
            "active_car": round(self.active_car, 8),
            "active_mdd": round(self.active_mdd, 8),
            "active_downside": round(self.active_downside, 8),
            "industry_cum_return": round(self.industry_cum_return, 8),
            "market_cum_return": round(self.market_cum_return, 8),
            "t0": self.t0,
            "n_returns": self.n_returns,
        }


class RiskCenterData:
    def __init__(self, config: RiskCenterConfig):
        self.config = config
        self.result_df = pd.read_csv(config.result_csv)
        self.result_df["event_date"] = self.result_df["event_date"].astype(str).str[:10]
        self.result_df["industry_code"] = self.result_df["industry_code"].astype(str)

        self._load_benchmark()
        self._load_market_and_industry_returns()
        self._load_optional_macro()

    def _load_benchmark(self) -> None:
        car = pd.read_csv(self.config.car_csv)
        car["event_date"] = car["event_date"].astype(str).str[:10]
        car["industry_code"] = car["industry_code"].astype(str)
        event_types = (
            car[car["window"] == 5][["event_date", "industry_code", "event_type"]]
            .drop_duplicates()
        )
        self.result_df = self.result_df.merge(
            event_types,
            on=["event_date", "industry_code"],
            how="left",
        )

    def _load_market_and_industry_returns(self) -> None:
        industry = pd.read_csv(self.config.industry_daily_csv)
        market = pd.read_csv(self.config.market_daily_csv)

        industry = industry.rename(
            columns={"日期": "date", "收盘": "close"}
        )
        market = market.rename(columns={"日期": "date", "收盘": "close"})
        industry["date"] = pd.to_datetime(industry["date"])
        market["date"] = pd.to_datetime(market["date"])
        industry["industry_code"] = industry["industry_code"].astype(str)

        industry = industry.sort_values(["industry_code", "date"])
        industry["return"] = industry.groupby("industry_code")["close"].pct_change(fill_method=None)
        industry = industry.dropna(subset=["return"])

        ind_wide = industry.pivot_table(
            index="date", columns="industry_code", values="return", aggfunc="last"
        ).sort_index()

        market = market.sort_values("date")
        market["return"] = market["close"].pct_change(fill_method=None)
        market = market.dropna(subset=["return"]).set_index("date")["return"].sort_index()

        common = sorted(set(ind_wide.index) & set(market.index))
        self.dates = pd.DatetimeIndex(common)
        self.industry_returns = ind_wide.loc[self.dates].astype(float)
        self.market_returns = market.loc[self.dates].astype(float)
        self.industry_names = {
            str(code): str(name)
            for code, name in industry[["industry_code", "industry_name"]].drop_duplicates().values
        }

    def _load_optional_macro(self) -> None:
        self.lpr = None
        self.margin = None
        self.fundamentals = None
        try:
            self.lpr = pd.read_csv(self.config.lpr_csv)
        except Exception:
            pass
        try:
            self.margin = pd.read_csv(self.config.margin_csv)
        except Exception:
            pass
        try:
            self.fundamentals = pd.read_csv(self.config.fundamentals_csv)
        except Exception:
            pass

    @property
    def industries(self) -> List[str]:
        return list(self.industry_returns.columns)

    def experiment_rows(self) -> pd.DataFrame:
        return self.result_df.copy()

    def realized_path(
        self,
        industry_code: str,
        event_date: str,
        horizon: int,
    ) -> Optional[RealizedPath]:
        industry_code = str(industry_code)
        if industry_code not in self.industry_returns.columns or horizon <= 0:
            return None

        event_dt = pd.Timestamp(str(event_date)[:10])
        start = int(self.dates.searchsorted(event_dt, side="left"))
        end = start + int(horizon)
        if start >= len(self.dates) or end > len(self.dates):
            return None

        ind_rets = self.industry_returns.iloc[start:end][industry_code].fillna(0.0).to_numpy(dtype=float)
        mkt_rets = self.market_returns.iloc[start:end].fillna(0.0).to_numpy(dtype=float)
        active_rets = (1.0 + ind_rets) / (1.0 + mkt_rets) - 1.0
        active_rets = np.nan_to_num(active_rets, nan=0.0, posinf=0.0, neginf=0.0)

        active_path = np.cumprod(1.0 + active_rets) - 1.0
        nav = 1.0 + active_path
        roll_max = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
        dd = (roll_max - nav) / np.maximum(roll_max, 1e-12)

        industry_cum = float(np.prod(1.0 + ind_rets) - 1.0)
        market_cum = float(np.prod(1.0 + mkt_rets) - 1.0)
        active_car = float(active_path[-1]) if len(active_path) else 0.0
        return RealizedPath(
            active_car=active_car,
            active_mdd=float(np.clip(np.max(dd), 0.0, 1.0)),
            active_downside=max(-active_car, 0.0),
            industry_cum_return=industry_cum,
            market_cum_return=market_cum,
            t0=str(self.dates[start].date()),
            n_returns=int(len(active_rets)),
        )

    def pre_event_beta(
        self,
        industry_code: str,
        event_date: str,
        window: Optional[int] = None,
    ) -> Tuple[float, bool]:
        """OLS beta of industry returns on market returns before event_date."""
        industry_code = str(industry_code)
        window = window or self.config.beta_window
        event_dt = pd.Timestamp(str(event_date)[:10])
        prior = self.dates[self.dates < event_dt]
        if len(prior) == 0 or industry_code not in self.industry_returns.columns:
            return 1.0, True

        sample_dates = prior[-int(window):]
        x = self.market_returns.loc[sample_dates].to_numpy(dtype=float)
        y = self.industry_returns.loc[sample_dates, industry_code].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < max(10, int(window * 0.3)):
            return 1.0, True
        x, y = x[mask], y[mask]
        beta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
        if not np.isfinite(beta):
            return 1.0, True
        return beta, False

    def industry_betas(self, as_of_date: str, window: Optional[int] = None) -> Dict[str, float]:
        window = window or self.config.beta_window
        as_of = pd.Timestamp(str(as_of_date)[:10])
        prior = self.dates[self.dates < as_of]
        if len(prior) == 0:
            return {str(c): 1.0 for c in self.industry_returns.columns}
        sample = prior[-int(window):]
        out: Dict[str, float] = {}
        for industry in self.industry_returns.columns:
            y = self.industry_returns.loc[sample, industry].to_numpy(dtype=float)
            x = self.market_returns.loc[sample].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if int(mask.sum()) < max(10, int(window * 0.3)):
                out[str(industry)] = 1.0
                continue
            beta = float(np.cov(x[mask], y[mask], ddof=1)[0, 1] / np.var(x[mask], ddof=1))
            out[str(industry)] = beta if np.isfinite(beta) else 1.0
        return out

    def historical_scenario_returns(
        self,
        start_date: str,
        end_date: str,
    ) -> Dict[str, float]:
        start = pd.Timestamp(str(start_date)[:10])
        end = pd.Timestamp(str(end_date)[:10])
        mask = (self.dates >= start) & (self.dates <= end)
        dates = self.dates[mask]
        out: Dict[str, float] = {}
        for industry in self.industry_returns.columns:
            rets = self.industry_returns.loc[dates, industry].fillna(0.0).to_numpy(dtype=float)
            out[str(industry)] = float(np.prod(1.0 + rets) - 1.0)
        market_rets = self.market_returns.loc[dates].fillna(0.0).to_numpy(dtype=float)
        out["HS300"] = float(np.prod(1.0 + market_rets) - 1.0)
        return out
