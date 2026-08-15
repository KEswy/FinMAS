"""
LOO experiment with v2 hierarchical-shrinkage prior.
=====================================================

Tests whether the new hierarchical-shrinkage prior architecture mitigates
the v1 LOO collapse (64.77% → 46.59%, drop of 18 pp).

Mechanism: under LOO, removing one event drops some (e, i) cells from n=3
to n=2. v1 reverts those to the default 1.0 (no scaling). v2 instead pools
information across event_type and industry, so the matrix coverage stays
~32 entries even after removing any single event.

Target outcome: v2 LOO >= 55-58% (closing the gap to v2 main, which should
land near 64% if the architecture is sound).
"""
import os, sys, datetime
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

from scripts.calibrate_sensitivity_v2 import compute_sensitivity_v2

# ── Pre-load full archive once (LRU caches inside calibrate_sensitivity_v2) ──
_DF_FULL = pd.read_csv("data/processed/car_results.csv")
_DF5     = _DF_FULL[_DF_FULL["window"] == 5].copy()
_DF5["industry_code"] = _DF5["industry_code"].astype(str)
_DF5["event_date"]    = _DF5["event_date"].astype(str).str[:10]


def v2_loo_matrix(exclude_event_date: str) -> dict:
    """Compute v2 sensitivity matrix excluding all CAR rows of given event date."""
    filtered = _DF5[_DF5["event_date"] != str(exclude_event_date)[:10]]
    return compute_sensitivity_v2(filtered, n_full=10)


# ── Tee logging ───────────────────────────────────────────────────────
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


TAG = "v2_loo"
TS  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"data/results/experiment_qwen_{TAG}_{TS}.txt"
out_path = f"data/processed/all_experiment_results_{TAG}_{TS}.csv"
tee = Tee(log_path); sys.stdout = tee

print(f"[v2 LOO] log: {log_path}")
print(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"投票次数: 1")
print("=" * 60)

# Show baseline v2 matrix on full data
import main as _main
ORIG_V1 = dict(_main.SECTOR_SENSITIVITY)
v2_full = compute_sensitivity_v2(_DF5, n_full=10)
print(f"\n[v1 baseline] {len(ORIG_V1)} entries (P_pos binning, n_min=3)")
print(f"[v2 full]     {len(v2_full)} entries (hierarchical shrinkage, n_full=10)")
print()

# Read targets
car_df = pd.read_csv("data/processed/car_results.csv")
targets = (
    car_df[car_df["window"] == 5]
    [["event_date", "event_name", "industry_code", "direction", "CAR"]]
    .drop_duplicates().reset_index(drop=True)
)
print(f"\n共 {len(targets)} 个 LOO 评测场景\n")

results = []
for idx, row in targets.iterrows():
    event_date    = str(row["event_date"])[:10]
    industry_code = str(row["industry_code"])
    event_name    = row["event_name"]
    real_car      = float(row["CAR"])
    real_dir      = "+" if real_car >= 0 else "-"

    # Per-event LOO matrix
    loo_matrix = v2_loo_matrix(event_date)
    _main.SECTOR_SENSITIVITY = loo_matrix

    print(f"\n[{idx+1}/{len(targets)}] {event_date} | {industry_code} | {event_name[:40]}")
    print(f"  [v2 LOO matrix] {len(loo_matrix)} entries (excluding event_date={event_date})")

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
        bt_v2 = r.get("backtest_v2_magnitude", {}) or {}
        bt_v3 = r.get("backtest_v3_magnitude_stoploss", {}) or {}
        bt_c10 = r.get("backtest_cost_010", {}) or {}
        bt_c20 = r.get("backtest_cost_020", {}) or {}
        bt_c30 = r.get("backtest_cost_030", {}) or {}

        results.append({
            "event_date": event_date, "event_name": event_name[:35],
            "industry_code": industry_code, "real_CAR": real_car,
            "real_dir": real_dir, "final_dir": pred_dir, "dir_correct": correct,
            "MAE": mae, "RMSE": r["metrics"]["prediction"]["RMSE"],
            "DirectionAcc": r["metrics"]["prediction"]["DirectionAcc"],
            "Coverage_95": r["metrics"]["prediction"]["Coverage_95"],
            "Sharpe": r["metrics"]["backtest"]["SharpeRatio"],
            "WinRate": r["metrics"]["backtest"]["WinRate"],
            "triggered": r["triggered_correction"],
            "ablation_tag": TAG,
            "loo_matrix_n": len(loo_matrix),
            # ── Disagreement-as-feature columns (Plan D) ──
            "ts_pre_signal":     dis.get("ts_pre_signal", ""),
            "ts_pre_pred5_mean": dis.get("ts_pre_pred5_mean", 0.0),
            "llm_signal":        dis.get("llm_signal", ""),
            "net_signal_inject": dis.get("net_signal_inject_value", 0.0),
            "disagreement":      dis.get("disagreement", False),
            # ── Position-sizing Sharpe variants (Plan Sharpe 修复) ──
            "Sharpe_v2_magnitude":    bt_v2.get("SharpeRatio", 0.0),
            "WinRate_v2_magnitude":   bt_v2.get("WinRate", 0.0),
            "Sharpe_v3_magstoploss":  bt_v3.get("SharpeRatio", 0.0),
            "WinRate_v3_magstoploss": bt_v3.get("WinRate", 0.0),
            # ── Cost sensitivity (Plan C, Stage 1.3) ────────────────
            "Sharpe_cost_010":  bt_c10.get("SharpeRatio", 0.0),
            "Sharpe_cost_020":  bt_c20.get("SharpeRatio", 0.0),
            "Sharpe_cost_030":  bt_c30.get("SharpeRatio", 0.0),
            **kp_results,
        })
    except Exception as e:
        print(f"  ✗ 失败: {e}")

res_df = pd.DataFrame(results)
res_df.to_csv(out_path, index=False, encoding="utf-8-sig")

print("\n" + "="*60)
print(f"v2 LOO 完成")
print("="*60)
if len(res_df) > 0:
    acc = res_df["dir_correct"].mean()
    print(f"\nCAR 符号准确率: {acc:.2%} ({res_df['dir_correct'].sum()}/{len(res_df)})")
    print(f"\n对比基准:")
    print(f"  v1 main (P_pos binning, full data):  64.77%")
    print(f"  v1 LOO  (P_pos binning, n_min=2):    46.59%  ← prior-collapse 暴露")
    print(f"  v2 LOO  (hierarchical shrinkage):    {acc:.2%}")
    delta_v1_main  = acc - 0.6477
    delta_v1_loo   = acc - 0.4659
    print(f"\n  vs v1 main 差值: {delta_v1_main:+.2%}")
    print(f"  vs v1 LOO  差值: {delta_v1_loo:+.2%}")
    if acc >= 0.58:
        print(f"  ✅ v2 LOO ≥ 58%，新 prior 架构成功消除大部分 prior 泄漏依赖")
    elif acc >= 0.55:
        print(f"  ✅ v2 LOO 在 55-58%，过硬底线，论文可写鲁棒性结果")
    elif acc >= 0.50:
        print(f"  ⚠️  v2 LOO 50-55%，比 v1 LOO 改善但仍显著低于 main")
    else:
        print(f"  ❌ v2 LOO < 50%，hierarchical shrinkage 也救不回，需要更激进的方案")

    pos = res_df[res_df["real_dir"] == "+"]
    neg = res_df[res_df["real_dir"] == "-"]
    print(f"\n正向事件: {pos['dir_correct'].mean():.2%}  负向事件: {neg['dir_correct'].mean():.2%}")
    print(f"平均 MAE: {res_df['MAE'].mean():.6f}")

    print(f"\n关键时点累积方向:")
    for k in (1, 3, 5, 10):
        col = f"CAR_dir_T{k}"
        if col in res_df.columns:
            valid = res_df[res_df[col].notna()]
            if len(valid) > 0:
                a = valid[col].astype(bool).mean()
                print(f"  T+{k}: {a:.2%}")

    car_t5 = car_df[car_df["window"] == 5][
        ["event_date", "industry_code", "event_type"]].copy()
    car_t5["event_date"] = car_t5["event_date"].astype(str).str[:10]
    car_t5["industry_code"] = car_t5["industry_code"].astype(str)
    res_df["event_date"] = res_df["event_date"].astype(str).str[:10]
    res_df["industry_code"] = res_df["industry_code"].astype(str)
    merged = res_df.merge(car_t5, on=["event_date", "industry_code"], how="left")
    if "event_type" in merged.columns:
        print(f"\n各事件类型准确率:")
        print(merged.groupby("event_type")["dir_correct"].agg(["mean", "count"]))

print(f"\nCSV: {out_path}")
print(f"日志: {log_path}")

sys.stdout = tee.terminal
tee.close()
print(f"\n✅ 完成: {log_path}")
