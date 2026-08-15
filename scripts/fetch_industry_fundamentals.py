"""
Fetch Shenwan first-tier INDUSTRY-LEVEL fundamentals (PE / PB / dividend yield)
via akshare, for v4 Phase-2 part3 (A path).

`index_analysis_daily_sw(symbol='一级行业', start, end)` returns all SW first-tier
industries but paginates day-by-day internally (~1030 trading days over
2019-2026, ~5s/day → ~1.5h) and occasionally returns a bad body (KeyError
'data'). One monolithic call is therefore fragile.

Fix: fetch PER YEAR with retry+backoff, caching each year to
data/raw/_fund_cache/<year>.csv. A crash or a bad-body year only re-runs THAT
year; completed years are skipped on re-run (resumable). Finally merge all
years, filter to our 21 codes, normalize → data/raw/industry_fundamentals.csv.

Daily series → firewall-clean (FundamentalsSkill slices strictly < event_date).

Run (v4 dir, conda env with akshare). Safe to re-run — it resumes:
    python scripts/fetch_industry_fundamentals.py
"""
import os, sys, time
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

import pandas as pd

try:
    import akshare as ak
except ImportError:
    print("akshare 未安装：pip install akshare", file=sys.stderr); sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(ROOT)
RAW_DIR = "data/raw"
CACHE_DIR = f"{RAW_DIR}/_fund_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

SW_CODES = {
    "801010": "农林牧渔", "801030": "化工", "801040": "钢铁",
    "801050": "有色金属", "801080": "电子", "801110": "家用电器",
    "801120": "食品饮料", "801150": "医药生物", "801160": "公用事业",
    "801170": "交通运输", "801180": "房地产", "801710": "建筑材料",
    "801730": "电气设备", "801750": "计算机", "801760": "传媒",
    "801770": "通信", "801780": "银行", "801790": "非银金融",
    "801880": "汽车", "801890": "机械设备",
}
COL_MAP = {
    "发布日期": "date", "指数代码": "industry_code",
    "市盈率": "pe_ttm", "市净率": "pb", "股息率": "dividend_yield",
}
YEARS = list(range(2019, 2027))
MAX_RETRY = 4


def fetch_year(year: int) -> pd.DataFrame | None:
    cache = f"{CACHE_DIR}/{year}.csv"
    if os.path.exists(cache):
        d = pd.read_csv(cache, dtype={"industry_code": str})
        print(f"  [{year}] 缓存命中 {len(d)} 行，跳过")
        return d
    start, end = f"{year}0101", f"{year}1231"
    for attempt in range(1, MAX_RETRY + 1):
        try:
            df = ak.index_analysis_daily_sw(symbol="一级行业",
                                            start_date=start, end_date=end)
            if df is None or len(df) == 0:
                raise ValueError("空返回")
            df = df.rename(columns=COL_MAP)
            df["industry_code"] = df["industry_code"].astype(str)
            df = df[df["industry_code"].isin(SW_CODES)].copy()
            keep = [c for c in ("date", "industry_code", "pe_ttm", "pb",
                                "dividend_yield") if c in df.columns]
            df = df[keep]
            df.to_csv(cache, index=False, encoding="utf-8-sig")
            print(f"  [{year}] ✓ {len(df)} 行 (attempt {attempt})")
            return df
        except Exception as e:
            wait = 3 * attempt
            print(f"  [{year}] attempt {attempt}/{MAX_RETRY} 失败: {repr(e)[:50]} "
                  f"→ {wait}s 后重试")
            time.sleep(wait)
    print(f"  [{year}] ✗ 放弃（{MAX_RETRY} 次均失败）")
    return None


def main() -> int:
    print("拉取申万一级行业估值（分年 + 缓存 + 重试，可续跑）...")
    parts, failed = [], []
    for y in YEARS:
        d = fetch_year(y)
        if d is not None and len(d):
            parts.append(d)
        else:
            failed.append(y)
        time.sleep(0.5)

    if not parts:
        print("\n❌ 全部年份失败。稍后重跑（会跳过已缓存年份）。", file=sys.stderr)
        return 1

    out = pd.concat(parts, ignore_index=True)
    out["industry_code"] = out["industry_code"].astype(str)
    out["industry_name"] = out["industry_code"].map(SW_CODES)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out = (out[["date", "industry_code", "industry_name", "pe_ttm", "pb",
                "dividend_yield"]]
           .drop_duplicates(["date", "industry_code"])
           .sort_values(["industry_code", "date"]).reset_index(drop=True))

    path = f"{RAW_DIR}/industry_fundamentals.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 保存 {path}: {out['industry_code'].nunique()}/{len(SW_CODES)} 行业, "
          f"{len(out)} 行")
    print(f"   日期范围: {out['date'].min()} ~ {out['date'].max()}")
    vc = out.groupby("industry_code").size()
    print(f"   每行业行数: min={vc.min()} max={vc.max()} median={int(vc.median())}")
    if failed:
        print(f"   ⚠ 失败年份 {failed}（重跑本脚本会补齐，已成功年份走缓存）")
    miss = [c for c in SW_CODES if c not in set(out['industry_code'])]
    if miss:
        print(f"   ⚠ 未取到行业码: {miss}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
