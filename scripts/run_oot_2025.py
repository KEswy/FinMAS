"""
Out-of-time evaluation: calibrate prior on 2020-2024, test on 2025.
=====================================================================

Addresses the in-sample-leakage concern raised by the by-year accuracy
breakdown:
  - 2020-2024 in-sample accuracy: 50.82% (61 events)
  - 2025      in-sample accuracy: 77.78% (27 events)

The 27pp gap could be:
  (A) 2025 events are genuinely easier to predict (e.g., 2025 has the
      geopolitical/market_event classes which carry stronger directional
      signal), OR
  (B) the hierarchical hybrid prior in our Round-4 main experiment is
      calibrated on the full 88 events including 2025, so the 2025
      evaluation has in-sample leakage advantage.

This OOT script calibrates the prior ONLY on 2020-2024 events (which
contain zero geopolitical / market_event observations, so those classes
fall back to default sensitivity 1.0), then runs the 27 events of 2025
under that frozen prior. Comparing OOT-2025 accuracy to in-sample-2025
accuracy (77.78%) tells us whether the prior is doing real lifting or
just memorising.

Cost: 27 events × ~8 min/event (Qwen 32B) = ~3-4 hours.
Output: CSV at data/processed/oot_2025_<ts>.csv, log at data/results/.

Usage:
    cd c:\\PycharmProjects\\BUAA\\files\\financial_agent_system\\financial_agent_system_v3_complete\\financial_agent_system_v3
    $env:LLM_MODEL="qwen2.5:32b"; $env:SEED="42"; python scripts/run_oot_2025.py
"""
from __future__ import annotations
import os, sys, datetime
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

from agents.llm_client import get_llm_config

from scripts.calibrate_sensitivity_v2 import compute_sensitivity_v2

# ── Load CAR archive, split 2020-2024 (train) vs 2025 (test) ─────────
_df = pd.read_csv("data/processed/car_results.csv")
_df5 = _df[_df["window"] == 5].copy()
_df5["industry_code"] = _df5["industry_code"].astype(str)
_df5["event_date_str"] = _df5["event_date"].astype(str).str[:10]
_df5["year"] = pd.to_datetime(_df5["event_date_str"]).dt.year

train_df = _df5[_df5["year"] <= 2024].copy()
test_df  = _df5[_df5["year"] == 2025].copy()
print(f"OOT split: train (2020-2024) = {len(train_df)} events, "
      f"test (2025) = {len(test_df)} events")
print(f"  Train event_type distribution: {dict(train_df['event_type'].value_counts())}")
print(f"  Test  event_type distribution: {dict(test_df['event_type'].value_counts())}")

# ── Calibrate prior on train only ────────────────────────────────────
OOT_MATRIX = compute_sensitivity_v2(train_df, n_full=5,
                                       n_pool_min=2,
                                       n_dense=3,
                                       min_dense_pairs=2)
print(f"\nOOT prior (calibrated on 2020-2024 only): {len(OOT_MATRIX)} entries")
for (et, ic), s in sorted(OOT_MATRIX.items()):
    print(f"  ({et!r}, {ic!r}): {s:+.3f}")

# ── Set up logging + Tee ─────────────────────────────────────────────
class Tee:
    def __init__(self, p):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(p), exist_ok=True)
        self.file = open(p, "w", encoding="utf-8")
    def write(self, m):
        self.terminal.write(m); self.file.write(m); self.file.flush()
    def flush(self):
        self.terminal.flush(); self.file.flush()
    def isatty(self):
        return self.terminal.isatty()
    def close(self):
        self.file.close()

_MODEL_RAW = get_llm_config()["model"]
_MODEL_TAG = _MODEL_RAW.split(":")[0].replace("/", "_")
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"data/results/experiment_{_MODEL_TAG}_oot2025_{TS}.txt"
out_path = f"data/processed/all_experiment_results_{_MODEL_TAG}_oot2025_{TS}.csv"
tee = Tee(log_path); sys.stdout = tee

print(f"\n[OOT 2025] log: {log_path}")
print(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ── Patch SECTOR_SENSITIVITY before invoking run_analysis ────────────
import main as _main
ORIG = dict(_main.SECTOR_SENSITIVITY)
_main.SECTOR_SENSITIVITY = OOT_MATRIX
print(f"\n[OOT prior] {len(OOT_MATRIX)} entries (calibrated on 2020-2024 only)")
print(f"[OOT prior] 2025 geopolitical/market_event have NO prior coverage")
print(f"          (fallback to default sensitivity 1.0 via main.SECTOR_SENSITIVITY.get)")

# ── Run only 2025 events ─────────────────────────────────────────────
targets = (
    test_df[["event_date_str", "event_name", "industry_code", "direction", "CAR"]]
    .rename(columns={"event_date_str": "event_date"})
    .drop_duplicates().reset_index(drop=True)
)
print(f"\n共 {len(targets)} 个 OOT 评测场景 (2025 only)\n")

results = []
for idx, row in targets.iterrows():
    event_date    = str(row["event_date"])[:10]
    industry_code = str(row["industry_code"])
    event_name    = row["event_name"]
    real_car      = float(row["CAR"])
    real_dir      = "+" if real_car >= 0 else "-"

    print(f"\n[{idx+1}/{len(targets)}] {event_date} | {industry_code} | {event_name[:40]}")

    try:
        r = _main.run_analysis(
            event=event_name,
            entity_names=[event_name[:20], industry_code],
            task_type="policy_event",
            industry_code=industry_code,
            event_date=event_date,
            blind_mode=True,
        )
        pf = r["prediction"].get("point_forecast_real", r["prediction"]["point_forecast"])
        pred_mean = float(np.mean(pf[:5]))
        pred_dir = "+" if pred_mean > 0 else "-"
        mae = r["metrics"]["prediction"]["MAE"]
        correct = (pred_dir == real_dir)
        print(f"  → 预测={pred_dir}({pred_mean:+.5f})  真实={real_dir}({real_car:+.4f})  "
              f"MAE={mae:.5f}  {'✅' if correct else '❌'}")

        kps = r.get("keypoints", {}) or {}
        kp_results = {f"CAR_dir_T{k}": bool(kps.get(f"CAR_dir_correct_T{k}", False))
                      for k in (1,3,5,10) if f"CAR_dir_correct_T{k}" in kps}
        dis = r.get("disagreement", {}) or {}

        results.append({
            "event_date": event_date, "event_name": event_name[:35],
            "industry_code": industry_code, "real_CAR": real_car,
            "real_dir": real_dir, "final_dir": pred_dir, "dir_correct": correct,
            "MAE": mae, "RMSE": r["metrics"]["prediction"]["RMSE"],
            "DirectionAcc": r["metrics"]["prediction"]["DirectionAcc"],
            "Sharpe": r["metrics"]["backtest"]["SharpeRatio"],
            "WinRate": r["metrics"]["backtest"]["WinRate"],
            "ablation_tag": "oot2025",
            "ts_pre_signal":     dis.get("ts_pre_signal", ""),
            "llm_signal":        dis.get("llm_signal", ""),
            "net_signal_inject": dis.get("net_signal_inject_value", 0.0),
            "disagreement":      dis.get("disagreement", False),
            **kp_results,
        })
    except Exception as e:
        print(f"  ✗ 失败: {e}")

    # 增量保存
    if (idx + 1) % 5 == 0 and len(results) > 0:
        pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  [增量保存] {len(results)} 条已落盘")

res_df = pd.DataFrame(results)
res_df.to_csv(out_path, index=False, encoding="utf-8-sig")

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"OOT 2025 evaluation 完成")
print("=" * 70)
if len(res_df) > 0:
    acc = res_df["dir_correct"].mean()
    print(f"\nOOT 2025 CAR 符号准确率: {acc:.2%} ({res_df['dir_correct'].sum()}/{len(res_df)})")
    print(f"对比 in-sample 2025 (Round 4 main, prior 含 2025):  77.78%")
    delta = acc - 0.7778
    print(f"差值: {delta:+.2%}")

    if abs(delta) < 0.05:
        print(f"\n✅ OOT 几乎不变（差 {abs(delta)*100:.1f}pp）→ prior 不靠 2025 数据 leakage")
        print("   解释 A 成立：2025 事件本身更容易预测，prior 没在做无效记忆")
    elif delta < -0.15:
        print(f"\n❌ OOT 大幅下降（−{abs(delta)*100:.1f}pp）→ prior 严重依赖 in-sample 2025")
        print("   解释 B 成立：需要在论文里讨论 in-sample leakage 影响")
    else:
        print(f"\n⚠️ OOT 中等下降（−{abs(delta)*100:.1f}pp）→ 中间情况，需要进一步分析")

    # By event_type
    print("\nBy event_type:")
    car_t5 = _df5[["event_date_str", "industry_code", "event_type"]].copy()
    car_t5.columns = ["event_date", "industry_code", "event_type"]
    res_df["event_date"] = res_df["event_date"].astype(str).str[:10]
    res_df["industry_code"] = res_df["industry_code"].astype(str)
    car_t5["industry_code"] = car_t5["industry_code"].astype(str)
    merged = res_df.merge(car_t5, on=["event_date","industry_code"], how="left")
    g = merged.groupby("event_type")["dir_correct"].agg(["mean","count"])
    g.columns = ["acc","n"]
    g["acc"] = g["acc"]*100
    print(g.to_string(float_format=lambda x: f"{x:.2f}"))

    pos = res_df[res_df["real_dir"]=="+"]
    neg = res_df[res_df["real_dir"]=="-"]
    print(f"\n正向事件: {pos['dir_correct'].mean()*100:.2f}% ({pos['dir_correct'].sum()}/{len(pos)})")
    print(f"负向事件: {neg['dir_correct'].mean()*100:.2f}% ({neg['dir_correct'].sum()}/{len(neg)})")
    print(f"平均 MAE: {res_df['MAE'].mean():.6f}")

print(f"\nCSV: {out_path}")
print(f"日志: {log_path}")

sys.stdout = tee.terminal
tee.close()
print(f"\n✅ 完成: {log_path}")
