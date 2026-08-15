"""Unified feature store with strict pre-event temporal isolation.

All market, valuation, sentiment, and macro features are built from rows whose
date is strictly before the event date. The event date itself is never used to
compute input features.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _to_date(value) -> _dt.date:
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    return _dt.datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


class FeatureStore:
    """Loads raw FinMAS data and produces leak-safe feature dictionaries."""

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)
        self.industry = self._load_industry()
        self.market = self._load_market()
        self.valuation = self._load_valuation()
        self.margin = self._load_margin()
        self.lpr = self._load_lpr()
        self._tfidf = None
        self._tfidf_corpus: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------
    def _load_industry(self) -> pd.DataFrame:
        path = self.data_dir / "raw" / "industry_daily.csv"
        df = pd.read_csv(path)
        df["日期"] = pd.to_datetime(df["日期"])
        df["industry_code"] = df["industry_code"].astype(str)
        df = df.sort_values(["industry_code", "日期"]).reset_index(drop=True)
        return df

    def _load_market(self) -> pd.DataFrame:
        path = self.data_dir / "raw" / "hs300.csv"
        df = pd.read_csv(path)
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.sort_values("日期").reset_index(drop=True)
        return df

    def _load_valuation(self) -> pd.DataFrame:
        path = self.data_dir / "raw" / "industry_fundamentals.csv"
        if not path.exists():
            return pd.DataFrame(columns=["date", "industry_code", "pe_ttm", "pb", "dividend_yield"])
        df = pd.read_csv(path, dtype={"industry_code": str})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["industry_code", "date"]).reset_index(drop=True)
        return df

    def _load_margin(self) -> pd.DataFrame:
        path = self.data_dir / "raw" / "margin_daily.csv"
        if not path.exists():
            return pd.DataFrame(columns=["date", "total_fin"])
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def _load_lpr(self) -> pd.DataFrame:
        path = self.data_dir / "raw" / "lpr.csv"
        if not path.exists():
            return pd.DataFrame(columns=["date", "LPR1Y", "LPR5Y"])
        df = pd.read_csv(path)
        df = df.rename(columns={"TRADE_DATE": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Feature builders
    # ------------------------------------------------------------------
    def market_snapshot(self, industry_code: str, event_date: str, lookback: int = 250) -> Dict[str, float]:
        """Market-derived features strictly before event_date."""
        code = str(industry_code)
        ind = self.industry[self.industry["industry_code"] == code].copy()
        if ind.empty:
            return {}

        event_dt = pd.Timestamp(_to_date(event_date))
        ind = ind[ind["日期"] < event_dt].copy()
        if ind.empty:
            return {}

        ind = ind.sort_values("日期")
        mkt = self.market[self.market["日期"] < event_dt].copy().sort_values("日期")
        if mkt.empty:
            return {}

        ind = ind.merge(mkt[["日期", "收盘"]], on="日期", how="left", suffixes=("", "_mkt"))
        ind["ret"] = ind["收盘"].pct_change()
        ind["vol_chg"] = ind["成交量"].pct_change()
        ind["mkt_ret"] = ind["收盘_mkt"].pct_change()
        ind["volatility"] = ind["ret"].rolling(5).std()
        ind["ma5_gap"] = ind["收盘"].rolling(5).mean() / ind["收盘"] - 1.0
        ind["ma20_gap"] = ind["收盘"].rolling(20).mean() / ind["收盘"] - 1.0
        ind = ind.dropna(subset=["ret", "mkt_ret"])
        if ind.empty:
            return {}

        tail = ind.tail(int(lookback))
        ret = tail["ret"].to_numpy(dtype=float)
        mkt_ret = tail["mkt_ret"].to_numpy(dtype=float)
        vol = tail["volatility"].to_numpy(dtype=float)
        vol_chg = tail["vol_chg"].to_numpy(dtype=float)

        recent_ret = ret[-20:] if len(ret) >= 20 else ret
        recent_mkt = mkt_ret[-20:] if len(mkt_ret) >= 20 else mkt_ret
        recent_vol = vol[-20:] if len(vol) >= 20 else vol

        hist_vol = float(np.nanstd(ret) + 1e-8)
        recent_vol = float(np.nanstd(recent_vol) + 1e-8)
        momentum_20 = float(np.nansum(recent_ret))
        relative_20 = float(np.nansum(recent_ret) - np.nansum(recent_mkt))
        volume_trend = float(np.nanmean(vol_chg[-20:])) if len(vol_chg) >= 20 else 0.0

        return {
            "momentum_20": momentum_20,
            "relative_strength_20": relative_20,
            "recent_vol": recent_vol,
            "hist_vol": hist_vol,
            "vol_regime": float(recent_vol / hist_vol - 1.0),
            "volume_trend": volume_trend,
            "ma5_gap": float(tail["ma5_gap"].iloc[-1]) if tail["ma5_gap"].notna().any() else 0.0,
            "ma20_gap": float(tail["ma20_gap"].iloc[-1]) if tail["ma20_gap"].notna().any() else 0.0,
            "n_hist": int(len(tail)),
        }

    def valuation_snapshot(self, industry_code: str, event_date: str) -> Dict[str, float]:
        code = str(industry_code)
        if self.valuation.empty:
            return {}
        event_dt = pd.Timestamp(_to_date(event_date))
        sub = self.valuation[
            (self.valuation["industry_code"] == code) & (self.valuation["date"] < event_dt)
        ].dropna(subset=["pe_ttm"]).sort_values("date")
        if len(sub) < 60:
            return {}
        pe = sub["pe_ttm"].to_numpy(dtype=float)
        pe_now = float(pe[-1])
        pe_pct = float((pe < pe_now).mean())
        out: Dict[str, float] = {
            "pe_now": pe_now,
            "pe_percentile": pe_pct,
            "valuation_signal": float(-(pe_pct - 0.5) * 2.0),
            "valuation_confidence": float(abs(pe_pct - 0.5) * 2.0),
            "n_hist": int(len(pe)),
        }
        if "pb" in sub.columns and sub["pb"].notna().any():
            pb = sub["pb"].dropna().to_numpy(dtype=float)
            out["pb_now"] = float(pb[-1])
            out["pb_percentile"] = float((pb < pb[-1]).mean())
        if "dividend_yield" in sub.columns and sub["dividend_yield"].notna().any():
            out["dividend_yield"] = float(sub["dividend_yield"].dropna().iloc[-1])
        return out

    def sentiment_snapshot(self, event_date: str, mom_window: int = 20, hist_window: int = 250) -> Dict[str, float]:
        if self.margin.empty:
            return {}
        event_dt = pd.Timestamp(_to_date(event_date))
        past = self.margin[self.margin["date"] < event_dt].sort_values("date")
        if len(past) < mom_window + 30:
            return {}
        s = past["total_fin"].to_numpy(dtype=float)
        mom = pd.Series(s).pct_change(mom_window).to_numpy(dtype=float)
        cur = float(mom[-1])
        hist = mom[-hist_window:]
        hist = hist[~np.isnan(hist)]
        if len(hist) < 20 or not np.isfinite(cur):
            return {}
        pctile = float((hist < cur).mean())
        return {
            "margin_momentum_20": cur,
            "froth": float(2.0 * pctile - 1.0),
            "froth_percentile": pctile,
            "n_hist": int(len(past)),
        }

    def macro_snapshot(self, event_date: str) -> Dict[str, float]:
        if self.lpr.empty:
            return {}
        event_dt = pd.Timestamp(_to_date(event_date))
        past = self.lpr[self.lpr["date"] < event_dt].sort_values("date")
        if past.empty:
            return {}
        last = past.iloc[-1]
        out: Dict[str, float] = {}
        for col in ("LPR1Y", "LPR5Y", "RATE_1", "RATE_2"):
            if col in past.columns and pd.notna(last[col]):
                series = past[col].dropna()
                if not series.empty:
                    out[col.lower()] = float(series.iloc[-1])
                    if len(series) >= 2:
                        out[f"{col.lower()}_chg"] = float(series.iloc[-1] - series.iloc[-2])
        return out

    def event_text_features(self, text: str, fit: bool = False) -> Dict[str, float]:
        """Deterministic text features without relying on an external embedding."""
        text = str(text or "")
        if fit or self._tfidf is None:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer

                corpus = self._load_event_corpus()
                vec = TfidfVectorizer(
                    max_features=128,
                    token_pattern=r"(?u)\b\w+\b",
                    ngram_range=(1, 2),
                )
                vec.fit(corpus)
                self._tfidf = vec
                self._tfidf_corpus = corpus
            except Exception:
                return {}
        if self._tfidf is None:
            return {}
        matrix = self._tfidf.transform([text]).toarray()[0]
        if not matrix.any():
            return {}
        top_idx = np.argsort(matrix)[-16:][::-1]
        return {f"tfidf_{i}": float(matrix[idx]) for i, idx in enumerate(top_idx)}

    def _load_event_corpus(self) -> List[str]:
        path = self.data_dir / "events" / "event_library.csv"
        if not path.exists():
            return [""]
        df = pd.read_csv(path)
        names = df["event_name"].astype(str).tolist()
        descs = df["description"].astype(str).tolist()
        return [f"{n} {d}" for n, d in zip(names, descs)]

    # ------------------------------------------------------------------
    # Public aggregate API
    # ------------------------------------------------------------------
    def build_features(
        self,
        industry_code: str,
        event_date: str,
        event_text: str = "",
        include_text: bool = True,
    ) -> Dict[str, float]:
        """Build a flat feature dict for one pre-event scenario."""
        features: Dict[str, float] = {}
        for snapshot in (
            self.market_snapshot(industry_code, event_date),
            self.valuation_snapshot(industry_code, event_date),
            self.sentiment_snapshot(event_date),
            self.macro_snapshot(event_date),
        ):
            features.update(snapshot)
        if include_text and event_text:
            features.update(self.event_text_features(event_text))
        return features

    def pre_event_matrix(self, industry_code: str, event_date: str, lookback: int = 250) -> Optional[pd.DataFrame]:
        """Return a feature matrix for time-series models, strictly pre-event."""
        code = str(industry_code)
        event_dt = pd.Timestamp(_to_date(event_date))
        ind = self.industry[self.industry["industry_code"] == code].copy()
        mkt = self.market.copy()
        if ind.empty or mkt.empty:
            return None
        ind = ind[ind["日期"] < event_dt].sort_values("日期")
        mkt = mkt[mkt["日期"] < event_dt].sort_values("日期")
        if ind.empty or mkt.empty:
            return None

        df = ind.merge(mkt[["日期", "收盘"]], on="日期", how="left", suffixes=("", "_mkt"))
        df["ret"] = df["收盘"].pct_change()
        df["vol_chg"] = df["成交量"].pct_change()
        df["volatility"] = df["ret"].rolling(5).std()
        df["mkt_ret"] = df["收盘_mkt"].pct_change()
        df["ma5_gap"] = df["收盘"].rolling(5).mean() / df["收盘"] - 1.0
        df["ma20_gap"] = df["收盘"].rolling(20).mean() / df["收盘"] - 1.0
        cols = ["ret", "vol_chg", "volatility", "mkt_ret", "ma5_gap", "ma20_gap"]
        df = df[["日期"] + cols].dropna()
        df = df.set_index("日期").astype("float32")
        if len(df) < lookback:
            return df
        return df.tail(lookback)

