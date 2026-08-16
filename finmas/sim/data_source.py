"""Market data source with akshare optional and local CSV fallback."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .schemas import MarketTick


class MarketDataSource:
    def __init__(self, data_dir: str = "data", cache_dir: str = "data/cache/sim") -> None:
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_live = os.environ.get("FINMAS_SIM_USE_LIVE", "0") == "1"

    def _cache(self, name: str, loader):
        path = self.cache_dir / f"{name}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        value = loader()
        path.write_text(
            json.dumps(value, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return value

    def _local_industry_returns(self, start: str, end: str) -> pd.DataFrame:
        industry = pd.read_csv(self.data_dir / "raw" / "industry_daily.csv")
        industry["日期"] = pd.to_datetime(industry["日期"])
        industry["industry_code"] = industry["industry_code"].astype(str)
        industry = industry.sort_values(["industry_code", "日期"])
        industry["ret"] = industry.groupby("industry_code")["收盘"].pct_change()
        wide = industry.pivot_table(
            index="日期", columns="industry_code", values="ret", aggfunc="last"
        ).sort_index()
        mask = (wide.index >= pd.Timestamp(start)) & (wide.index <= pd.Timestamp(end))
        return wide.loc[mask].fillna(0.0)

    def _local_market_returns(self, start: str, end: str) -> pd.Series:
        market = pd.read_csv(self.data_dir / "raw" / "hs300.csv")
        market["日期"] = pd.to_datetime(market["日期"])
        market["ret"] = market["收盘"].pct_change()
        market = market.dropna().set_index("日期")["ret"].sort_index()
        mask = (market.index >= pd.Timestamp(start)) & (market.index <= pd.Timestamp(end))
        return market.loc[mask].fillna(0.0)

    def load_industry_returns(self, start: str, end: str) -> pd.DataFrame:
        try:
            if not self.use_live:
                raise RuntimeError("live disabled")
            import akshare as ak

            frames = {}
            for code in [
                "801010", "801030", "801040", "801050", "801080",
                "801110", "801120", "801130", "801140", "801150",
                "801160", "801170", "801180", "801200", "801210",
                "801230", "801710", "801720", "801730", "801740",
                "801750", "801760", "801770", "801780", "801790",
                "801880", "801890", "801950",
            ]:
                df = ak.index_hist_sw(symbol=code)
                df = df.rename(columns={"日期": "date", "收盘": "close"})
                df["date"] = pd.to_datetime(df["date"])
                df["industry_code"] = code
                df["return"] = df.groupby("industry_code")["close"].pct_change()
                frames[code] = df[["date", "industry_code", "return"]].dropna()
            all_frames = pd.concat(frames.values(), ignore_index=True)
            wide = all_frames.pivot_table(
                index="date", columns="industry_code", values="return", aggfunc="last"
            ).sort_index()
            mask = (wide.index >= pd.Timestamp(start)) & (wide.index <= pd.Timestamp(end))
            return wide.loc[mask].fillna(0.0)
        except Exception:
            return self._local_industry_returns(start, end)

    def load_market_returns(self, start: str, end: str) -> pd.Series:
        try:
            if not self.use_live:
                raise RuntimeError("live disabled")
            import akshare as ak

            df = ak.stock_zh_index_daily(symbol="sh000300")
            df = df.rename(columns={"date": "date", "close": "close"})
            df["date"] = pd.to_datetime(df["date"])
            df["return"] = df["close"].pct_change()
            df = df.dropna().set_index("date")["return"].sort_index()
            mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
            return df.loc[mask].fillna(0.0)
        except Exception:
            return self._local_market_returns(start, end)

    def load_capital_flow(self, start: str, end: str) -> pd.Series:
        if self.use_live:
            try:
                import akshare as ak

                df = ak.stock_margin_sse(
                    start_date=pd.Timestamp(start).strftime("%Y%m%d"),
                    end_date=pd.Timestamp(end).strftime("%Y%m%d"),
                )
                df = df.rename(columns={"信用交易日期": "date", "融资余额": "balance"})
                df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
                df = df.sort_values("date").set_index("date")["balance"]
                return df.pct_change().fillna(0.0)
            except Exception:
                pass
        margin = pd.read_csv(self.data_dir / "raw" / "margin_daily.csv")
        margin["date"] = pd.to_datetime(margin["date"])
        margin = margin.sort_values("date").set_index("date")["total_fin"]
        margin = margin.pct_change().fillna(0.0)
        mask = (margin.index >= pd.Timestamp(start)) & (margin.index <= pd.Timestamp(end))
        return margin.loc[mask]

    def tick(self, date: str) -> MarketTick:
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        start = end = date_str
        market = self.load_market_returns(start, end)
        industries = self.load_industry_returns(start, end)
        flow = self.load_capital_flow(start, end)
        market_return = float(market.iloc[0]) if len(market) else 0.0
        industry_returns = (
            industries.iloc[0].to_dict() if len(industries) else {}
        )
        capital_flow = {
            "margin_change": float(flow.iloc[0]) if len(flow) else 0.0
        }
        return MarketTick(
            date=date_str,
            market_return=market_return,
            industry_returns={str(k): float(v) for k, v in industry_returns.items()},
            capital_flow=capital_flow,
        )
