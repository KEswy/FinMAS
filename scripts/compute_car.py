"""
计算行业累计异常收益率（CAR）
窗口：事件日后 t+1, t+5, t+20 交易日
CAR = 行业累计收益 - 市场基准累计收益
"""
import pandas as pd
import numpy as np
import os

RAW_DIR    = "../data/raw"
PROC_DIR   = "../data/processed"
EVENTS_DIR = "../data/events"
os.makedirs(PROC_DIR, exist_ok=True)

# ── 1. 读取行业数据 ──────────────────────────
print("读取行业数据...")
industry_df = pd.read_csv(f"{RAW_DIR}/industry_daily.csv")
industry_df["日期"] = pd.to_datetime(industry_df["日期"])
print(f"  行业数据：{len(industry_df)} 条，列名：{industry_df.columns.tolist()}")

# ── 2. 读取 hs300，自动识别列名 ──────────────
print("\n读取沪深300基准...")
hs300_raw = pd.read_csv(f"{RAW_DIR}/hs300.csv")
print(f"  hs300列名：{hs300_raw.columns.tolist()}")
print(f"  hs300前2行：\n{hs300_raw.head(2)}")

# 自动识别日期列和收盘列
date_col  = next((c for c in hs300_raw.columns
                  if any(k in str(c).lower() for k in ["date","日期","time"])), hs300_raw.columns[0])
close_col = next((c for c in hs300_raw.columns
                  if any(k in str(c).lower() for k in ["close","收盘"])), None)
if close_col is None:
    # 取第一个数值列
    close_col = next(c for c in hs300_raw.columns
                     if hs300_raw[c].dtype in [float, "float64","float32"])

print(f"  识别日期列='{date_col}'，收盘列='{close_col}'")

hs300 = hs300_raw.copy()
hs300["日期"]  = pd.to_datetime(hs300[date_col])
hs300["收盘"]  = pd.to_numeric(hs300[close_col], errors="coerce")
hs300 = hs300[["日期","收盘"]].sort_values("日期").set_index("日期")
hs300["mkt_ret"] = hs300["收盘"].pct_change(fill_method=None)
print(f"  沪深300日期范围：{hs300.index.min().date()} ~ {hs300.index.max().date()}")

# ── 3. 构建行业收益率宽表 ────────────────────
print("\n计算行业收益率...")
industry_pivot = industry_df.pivot_table(
    index="日期", columns="industry_code", values="收盘", aggfunc="mean"
).sort_index()
industry_ret = industry_pivot.pct_change(fill_method=None)
industry_ret.columns = industry_ret.columns.astype(str)  # 统一为字符串
print(f"  行业数：{len(industry_ret.columns)}，日期范围：{industry_ret.index.min().date()} ~ {industry_ret.index.max().date()}")

# ── 4. 获取有效交易日列表 ────────────────────
# 取行业数据与hs300的交集交易日
trading_days = sorted(set(industry_ret.index) & set(hs300.index))
print(f"  有效交易日数：{len(trading_days)}")

# ── 5. 读取事件库 ────────────────────────────
print("\n读取事件库...")
events = pd.read_csv(f"{EVENTS_DIR}/event_library.csv")
events["event_date"] = pd.to_datetime(events["event_date"])
print(f"  事件数：{len(events)}")

# ── 6. 对每个事件计算 CAR ────────────────────
print("\n计算CAR...")
all_car_records = []

for _, event in events.iterrows():
    event_date = event["event_date"]
    affected   = [c.strip() for c in str(event["affected_industries"]).split(",")]

    # 找到事件日当天或之后的第一个交易日作为 t0
    future_days = [d for d in trading_days if d >= event_date]
    if not future_days:
        print(f"  ✗ {event['event_name'][:25]}：事件日{event_date.date()}之后无交易日")
        continue
    t0     = future_days[0]
    t0_idx = trading_days.index(t0)

    for window_end in [1, 5, 20]:
        end_idx = t0_idx + window_end
        if end_idx >= len(trading_days):
            continue

        window_days = trading_days[t0_idx: end_idx + 1]

        # 市场基准 CAR
        mkt_rets = hs300.loc[hs300.index.isin(window_days), "mkt_ret"].fillna(0)
        cum_mkt  = float((1 + mkt_rets).prod() - 1)

        for ind_code in affected:
            if ind_code not in industry_ret.columns:
                continue
            ind_rets = industry_ret.loc[
                industry_ret.index.isin(window_days), ind_code
            ].fillna(0)
            cum_ind = float((1 + ind_rets).prod() - 1)
            car     = cum_ind - cum_mkt

            all_car_records.append({
                "event_date":    event_date.strftime("%Y-%m-%d"),
                "event_type":    event["event_type"],
                "event_name":    event["event_name"],
                "industry_code": ind_code,
                "direction":     event["direction"],
                "magnitude":     event["magnitude"],
                "window":        window_end,
                "t0":            t0.strftime("%Y-%m-%d"),
                "cum_ind_ret":   round(cum_ind, 6),
                "cum_mkt_ret":   round(cum_mkt, 6),
                "CAR":           round(car, 6),
            })

# ── 7. 保存结果 ──────────────────────────────
car_df = pd.DataFrame(all_car_records)

if len(car_df) == 0:
    print("\n❌ CAR为0条，调试信息：")
    print(f"  事件日期示例：{events['event_date'].head(3).tolist()}")
    print(f"  交易日示例：{trading_days[:3]}")
else:
    car_df.to_csv(f"{PROC_DIR}/car_results.csv",
                  index=False, encoding="utf-8-sig")
    print(f"\n✅ CAR计算完成，共 {len(car_df)} 条记录")

    print("\n各窗口平均CAR（所有事件）：")
    print(car_df.groupby("window")["CAR"].agg(["mean","std","count"]).round(4))

    print("\n正向事件各窗口平均CAR：")
    pos = car_df[car_df["direction"] == "+"]
    print(pos.groupby("window")["CAR"].agg(["mean","std","count"]).round(4))

    print("\nCAR最显著的前10个事件（t+5窗口）：")
    top = car_df[car_df["window"]==5].nlargest(10, "CAR")[
        ["event_date","event_name","industry_code","CAR"]
    ]
    print(top.to_string(index=False))