r"""
Sentiment skill (v4 Phase 5) -- market-level retail froth, strictly pre-t.
==========================================================================
Motivation: in A-shares, transmission theory is frequently overridden by retail
sentiment (a rate cut is theory-negative for bank NIM, yet banks often rally on
easing-driven risk appetite). We capture *market-level* leverage sentiment
(margin-financing balance = 融资余额) as a "froth" signal and use it to GATE the
IRF transmission prior: when the market is frothy, bearish fundamentals get
overridden, so we down-weight a bearish theory cell; when cold, theory reasserts.

Leakage safety:
  - MARKET-LEVEL only (SSE+SZSE totals) -> no per-stock -> industry mapping, so
    no time-varying-constituent leak (the trap that retired part3-B).
  - froth(t) reads ONLY margin rows with date < t. Truncating the frame at t
    leaves the value bit-identical (isolation test in test_sentiment()).

froth(t) in [-1, 1]:  +1 = extreme greed/leverage build-up, -1 = extreme
fear/deleveraging. Built from the percentile of 20d margin-balance momentum
within its own trailing (<t) distribution.
"""
from __future__ import annotations
import os
import datetime as _dt
from typing import Tuple
import pandas as pd
import numpy as np

_MARGIN_CSV = "data/raw/margin_daily.csv"
_MOM_WIN = int(os.environ.get("SENT_MOM_WIN", "20"))     # momentum window (trading days)
_HIST_WIN = int(os.environ.get("SENT_HIST_WIN", "250"))  # trailing window for percentile


def _to_date(x) -> _dt.date:
    if isinstance(x, _dt.date) and not isinstance(x, _dt.datetime):
        return x
    return _dt.datetime.strptime(str(x)[:10], "%Y-%m-%d").date()


class SentimentSkill:
    def __init__(self, margin_csv: str = _MARGIN_CSV):
        self._df = None
        self._csv = margin_csv

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            d = pd.read_csv(self._csv)
            d["date"] = pd.to_datetime(d["date"])
            d = d.sort_values("date").reset_index(drop=True)
            self._df = d
        return self._df

    def froth(self, event_dt) -> Tuple[float, dict]:
        """Market froth in [-1,1], strictly pre-t. (value, info)."""
        try:
            d = self._load()
            t = pd.Timestamp(_to_date(event_dt))
            past = d[d["date"] < t]                        # STRICT pre-t
            if len(past) < _MOM_WIN + 30:
                return 0.0, {"sentiment": "cold_start", "n_pre_t": int(len(past))}
            s = past["total_fin"].values
            # 20d leverage momentum series (each point uses only its own past)
            mom = pd.Series(s).pct_change(_MOM_WIN).values
            cur = mom[-1]
            hist = mom[-_HIST_WIN:] if len(mom) >= _HIST_WIN else mom
            hist = hist[~np.isnan(hist)]
            if hist.size < 20 or np.isnan(cur):
                return 0.0, {"sentiment": "thin", "n_pre_t": int(len(past))}
            pctile = float((hist < cur).mean())            # [0,1]
            froth = 2.0 * pctile - 1.0                      # [-1,1]
            return froth, {"sentiment": "ok", "n_pre_t": int(len(past)),
                           "mom20": round(float(cur), 4), "pctile": round(pctile, 3),
                           "froth": round(froth, 3)}
        except FileNotFoundError:
            return 0.0, {"sentiment": "no_data"}
        except Exception as e:
            return 0.0, {"sentiment": f"err:{type(e).__name__}"}


def test_sentiment():
    """Isolation smoke test: froth(t) must not change when future rows are added."""
    sk = SentimentSkill()
    for dt in ("2024-07-22", "2023-01-10", "2021-05-20"):
        v, info = sk.froth(dt)
        print(f"  froth({dt}) = {v:+.3f}  {info}")
    # isolation: truncate the frame at t, recompute -> identical
    import numpy as _np
    sk2 = SentimentSkill()
    d = sk2._load()
    t = pd.Timestamp("2024-07-22")
    full_v, _ = sk2.froth("2024-07-22")
    sk3 = SentimentSkill(); sk3._df = d[d["date"] < t].copy()
    trunc_v, _ = sk3.froth("2024-07-22")
    ok = abs(full_v - trunc_v) < 1e-12
    print(f"  isolation full={full_v:+.6f} trunc={trunc_v:+.6f} -> {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    test_sentiment()
