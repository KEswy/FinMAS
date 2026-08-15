r"""
Fetch market-level retail-sentiment data (margin financing) for the sentiment
skill. STRICTLY market-level (SSE + SZSE totals) -- no per-stock -> industry
mapping, so no time-varying-constituent leak (the trap that killed part3-B).

Output: data/raw/margin_daily.csv  with columns [date, sse_fin, szse_fin, total_fin]
  sse_fin  = SSE 融资余额 (margin-buy balance), yuan
  szse_fin = SZSE 融资余额
  total_fin= sse_fin + szse_fin  (whole-A margin balance = leverage/greed proxy)

Range 2019-06-01 .. today so the earliest event (2020-02-20) has >=60 trading
days of pre-t history for the froth trailing windows.

CLI:  python scripts/fetch_sentiment_data.py
"""
from __future__ import annotations
import os, sys, io, time
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import akshare as ak

START, END = "20190601", None  # END=None -> today


def _today():
    import datetime as dt
    return dt.datetime.now().strftime("%Y%m%d")


def _fetch_sse(start, end):
    """SSE margin: first numeric col after date is 融资余额. Column names vary by
    akshare version, so index by position (date, rzye, ...)."""
    df = ak.stock_margin_sse(start_date=start, end_date=end)
    df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "sse_fin"})
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df["sse_fin"] = pd.to_numeric(df["sse_fin"], errors="coerce")
    return df[["date", "sse_fin"]].dropna()


def _fetch_szse(start, end):
    """SZSE margin: akshare needs per-date query for szse; use the summary total
    series. stock_margin_szse(date=...) returns one day, so we pull the SSE date
    grid and query szse per date would be slow. Instead use macro market_margin_sz
    which is a full daily series (融资余额 col)."""
    df = ak.macro_china_market_margin_sz()
    # columns include 日期 + 融资余额(元) or similar; find date + a 融资余额 col
    cols = list(df.columns)
    date_col = cols[0]
    fin_col = None
    for c in cols:
        s = str(c)
        if "融资余额" in s and "融券" not in s:
            fin_col = c; break
    if fin_col is None:  # fallback: second column
        fin_col = cols[1]
    out = df.rename(columns={date_col: "date", fin_col: "szse_fin"})[["date", "szse_fin"]]
    out["date"] = pd.to_datetime(out["date"].astype(str), errors="coerce")
    out["szse_fin"] = pd.to_numeric(out["szse_fin"], errors="coerce")
    return out.dropna()


def main():
    end = END or _today()
    print(f"[sentiment] fetching margin balance {START}..{end}")
    for attempt in range(3):
        try:
            sse = _fetch_sse(START, end)
            print(f"  SSE rows={len(sse)}  {sse['date'].min().date()}..{sse['date'].max().date()}")
            break
        except Exception as e:
            print(f"  SSE retry {attempt+1}: {repr(e)[:120]}"); time.sleep(3)
    else:
        print("  SSE fetch failed"); return 1
    try:
        szse = _fetch_szse(START, end)
        print(f"  SZSE rows={len(szse)}  {szse['date'].min().date()}..{szse['date'].max().date()}")
    except Exception as e:
        print(f"  SZSE fetch failed ({repr(e)[:100]}); proceeding SSE-only")
        szse = pd.DataFrame(columns=["date", "szse_fin"])

    m = sse.merge(szse, on="date", how="left").sort_values("date")
    m["szse_fin"] = m["szse_fin"].ffill()
    m["total_fin"] = m["sse_fin"] + m["szse_fin"].fillna(0)
    os.makedirs("data/raw", exist_ok=True)
    out = "data/raw/margin_daily.csv"
    m.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[sentiment] wrote {out}  rows={len(m)}  "
          f"total_fin {m['total_fin'].iloc[0]:.3e}..{m['total_fin'].iloc[-1]:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
