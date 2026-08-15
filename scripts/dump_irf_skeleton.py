"""
Dump the (event_type, industry_code, industry_name) skeleton that the IRF
prior must cover — the exact rows×cols present in the benchmark. Also reports
per-cell historical CAR stats (sign rate + mean), which we use ONLY to
calibrate prior STRENGTH (direction comes from transmission theory). Read-only,
seconds, no GPU.

Run:  python scripts/dump_irf_skeleton.py
Output: prints a table + writes data/processed/irf_skeleton.csv
"""
from __future__ import annotations
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

car = pd.read_csv("data/processed/car_results.csv")
car = car[car["window"] == 5].copy()
car["industry_code"] = car["industry_code"].astype(str)

ind = pd.read_csv("data/raw/industry_daily.csv", usecols=["industry_code", "industry_name"])
ind["industry_code"] = ind["industry_code"].astype(str)
name_map = dict(ind.drop_duplicates("industry_code").values)

rows = []
for (et, ic), g in car.groupby(["event_type", "industry_code"]):
    n = len(g)
    ppos = float((g["CAR"] >= 0).mean())
    mean_car = float(g["CAR"].mean())
    rows.append({
        "event_type": et,
        "industry_code": ic,
        "industry_name": name_map.get(ic, "?"),
        "n": n,
        "P_pos": round(ppos, 3),
        "mean_CAR": round(mean_car, 4),
    })

df = pd.DataFrame(rows).sort_values(["event_type", "n"], ascending=[True, False])
out = "data/processed/irf_skeleton.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")

print(f"共 {len(df)} 个 (event_type, industry) cell，覆盖 "
      f"{df['event_type'].nunique()} 类事件 × {df['industry_code'].nunique()} 个行业")
print(f"\n各事件类型的 cell 数:")
print(df.groupby("event_type").size().to_string())
print(f"\n完整骨架:")
print(df.to_string(index=False))
print(f"\n已写: {out}")
