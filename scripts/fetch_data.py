import akshare as ak
import pandas as pd
import os, time

os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

RAW_DIR = "../data/raw"
os.makedirs(RAW_DIR, exist_ok=True)
START = "20190101"
END   = "20260413"

SW_CODES = {
    "801010": "农林牧渔", "801020": "采掘",
    "801030": "化工",    "801040": "钢铁",
    "801050": "有色金属", "801080": "电子",
    "801110": "家用电器", "801120": "食品饮料",
    "801150": "医药生物", "801160": "公用事业",
    "801170": "交通运输", "801180": "房地产",
    "801710": "建筑材料", "801730": "电气设备",
    "801750": "计算机",  "801760": "传媒",
    "801770": "通信",    "801780": "银行",
    "801790": "非银金融", "801880": "汽车",
    "801890": "机械设备",
}

# ── 1. 申万行业指数 ──────────────────────────
print("拉取申万行业指数（index_hist_sw）...")
all_industry = []

for code, name in SW_CODES.items():
    try:
        df = ak.index_hist_sw(symbol=code, period="day")
        df["日期"] = pd.to_datetime(df["日期"])
        df = df[(df["日期"] >= START) & (df["日期"] <= END)].copy()
        df["industry_code"] = code
        df["industry_name"] = name
        all_industry.append(df)
        print(f"  ✓ {name}（{len(df)}条）")
        time.sleep(0.3)
    except Exception as e:
        print(f"  ✗ {name}: {e}")

if all_industry:
    result = pd.concat(all_industry, ignore_index=True)
    result.to_csv(f"{RAW_DIR}/industry_daily.csv",
                  index=False, encoding="utf-8-sig")
    print(f"\n✅ 行业数据保存完成：{len(all_industry)} 个行业，{len(result)} 条\n")
else:
    print("\n❌ 全部失败\n")

# ── 2. 沪深300基准（多方法尝试）────────────────
print("拉取沪深300...")
hs300 = None

# 方法1: index_zh_a_hist（东方财富源，最稳定）
try:
    print("  尝试方法1: index_zh_a_hist ...")
    hs300 = ak.index_zh_a_hist(symbol="000300", period="daily",
                                start_date=START, end_date=END)
    # 统一列名：这个接口返回的列名是中文的"日期","收盘"等
    if "日期" not in hs300.columns:
        # 有些版本列名是英文
        col_map = {}
        for c in hs300.columns:
            cl = c.lower()
            if "date" in cl: col_map[c] = "日期"
            elif "close" in cl: col_map[c] = "收盘"
            elif "open" in cl: col_map[c] = "开盘"
            elif "high" in cl: col_map[c] = "最高"
            elif "low" in cl: col_map[c] = "最低"
            elif "volume" in cl: col_map[c] = "成交量"
            elif "amount" in cl or "turnover" in cl: col_map[c] = "成交额"
        if col_map:
            hs300 = hs300.rename(columns=col_map)
    hs300["日期"] = pd.to_datetime(hs300["日期"])
    print(f"  ✓ 方法1成功（{len(hs300)}条）")
except Exception as e:
    print(f"  ✗ 方法1失败: {e}")

# 方法2: stock_zh_index_daily（新浪源）
if hs300 is None or len(hs300) == 0:
    try:
        print("  尝试方法2: stock_zh_index_daily ...")
        hs300 = ak.stock_zh_index_daily(symbol="sh000300")
        # 这个接口返回英文列名: date, open, high, low, close, volume
        hs300 = hs300.rename(columns={
            "date": "日期", "close": "收盘", "open": "开盘",
            "high": "最高", "low": "最低", "volume": "成交量",
        })
        hs300["日期"] = pd.to_datetime(hs300["日期"])
        hs300 = hs300[(hs300["日期"] >= START) & (hs300["日期"] <= END)]
        print(f"  ✓ 方法2成功（{len(hs300)}条）")
    except Exception as e:
        print(f"  ✗ 方法2失败: {e}")

# 方法3: stock_zh_index_daily_em（东方财富源）
if hs300 is None or len(hs300) == 0:
    try:
        print("  尝试方法3: stock_zh_index_daily_em ...")
        hs300 = ak.stock_zh_index_daily_em(symbol="sh000300",
                                            start_date=START, end_date=END)
        if "日期" not in hs300.columns:
            col_map = {}
            for c in hs300.columns:
                cl = c.lower()
                if "date" in cl or "日期" in c: col_map[c] = "日期"
                elif "close" in cl or "收盘" in c: col_map[c] = "收盘"
                elif "open" in cl or "开盘" in c: col_map[c] = "开盘"
                elif "high" in cl or "最高" in c: col_map[c] = "最高"
                elif "low" in cl or "最低" in c: col_map[c] = "最低"
            if col_map:
                hs300 = hs300.rename(columns=col_map)
        hs300["日期"] = pd.to_datetime(hs300["日期"])
        print(f"  ✓ 方法3成功（{len(hs300)}条）")
    except Exception as e:
        print(f"  ✗ 方法3失败: {e}")

# 保存结果
if hs300 is not None and len(hs300) > 0:
    hs300.to_csv(f"{RAW_DIR}/hs300.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ 沪深300保存完成（{len(hs300)}条）")
    print(f"  列名: {hs300.columns.tolist()}")
    print(f"  日期范围: {hs300['日期'].min()} ~ {hs300['日期'].max()}")
else:
    print("\n❌ 沪深300全部拉取方法失败！")
    print("  请手动下载沪深300日线数据，确保包含'日期'和'收盘'列，")
    print(f"  保存到 {RAW_DIR}/hs300.csv")

# ── 3. 验证 hs300 列名与行业数据对齐 ─────────
print("\n验证数据格式...")
try:
    ind = pd.read_csv(f"{RAW_DIR}/industry_daily.csv")
    print(f"  行业数据列名: {ind.columns.tolist()}")
    print(f"  行业数: {ind['industry_code'].nunique()}")
    print(f"  日期范围: {ind['日期'].min()} ~ {ind['日期'].max()}")
except Exception as e:
    print(f"  行业数据读取失败: {e}")

try:
    hs = pd.read_csv(f"{RAW_DIR}/hs300.csv")
    print(f"  沪深300列名: {hs.columns.tolist()}")
    print(f"  沪深300条数: {len(hs)}")
except Exception as e:
    print(f"  沪深300读取失败: {e}")

# ── 4. LPR（已存在则跳过）────────────────────
if os.path.exists(f"{RAW_DIR}/lpr.csv"):
    print("\nLPR已存在，跳过")
else:
    try:
        lpr = ak.macro_china_lpr()
        lpr.to_csv(f"{RAW_DIR}/lpr.csv", index=False, encoding="utf-8-sig")
        print(f"\n✓ LPR保存完成（{len(lpr)}条）")
    except Exception as e:
        print(f"\n✗ LPR: {e}")

print("\n全部完成！")
