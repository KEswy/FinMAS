"""
Main experiment with v2 hierarchical-shrinkage prior.
======================================================

Sanity check: does the new prior architecture preserve (or improve) the
headline 64.77% CAR sign accuracy when given the FULL historical archive?

If v2 main >= 60% → architecture is sound, proceed to LOO test.
If v2 main < 55% → architecture is broken, revert to v1 + apply different
                   LOO mitigation (e.g. pooled fallback for n_min=2 cases).

The v2 matrix here uses the FULL car_results.csv (no event excluded), so
this is the headline-equivalent for the new prior. It is NOT the LOO
robustness number — that comes from run_all_events_v2_loo.py.
"""
import os, sys, datetime
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

from agents.llm_client import get_llm_config
from scripts.calibrate_sensitivity_v2 import compute_sensitivity_v2

# ── Pre-compute the v2 matrix from full car_results.csv ───────────────
df = pd.read_csv("data/processed/car_results.csv")
df5 = df[df["window"] == 5].copy()
df5["industry_code"] = df5["industry_code"].astype(str)
V2_MATRIX = compute_sensitivity_v2(df5, n_full=10)

# Tier M fix (2026-05-22): clamp negative cells to 0 to prevent double-negative
# sign-flip on event types where LLM already correctly identifies direction
# (market_event, geopolitical). Negative sensitivity meant "history overrules
# LLM direction" — useful at n=88 when most events were monetary_policy with
# positive LLM read, but at n=168 with 13 market_event + 22 geopolitical events
# (all LLM-correctly-negative), negative sensitivity × negative net_signal
# = positive correction, flipping direction. Sanity drop 59% → 51.79%.
# After clamp: sensitivity becomes pure magnitude scaler, LLM controls sign.
_n_neg = sum(1 for v in V2_MATRIX.values() if v < 0)
V2_MATRIX = {k: max(v, 0.0) for k, v in V2_MATRIX.items()}
_n_kept = sum(1 for v in V2_MATRIX.values() if v > 0)
print(f"[v2 prior clamp] zeroed {_n_neg} negative cells, kept {_n_kept} positive")

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

_SEED_TAG  = os.environ.get("SEED", "42")
_MODEL_RAW = get_llm_config()["model"]
_MODEL_TAG = _MODEL_RAW.split(":")[0].replace("/", "_")  # qwen2.5 / llama3.1 / deepseek-r1
TAG = f"v2_seed{_SEED_TAG}" if _SEED_TAG != "42" else "v2"
TS  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"data/results/experiment_{_MODEL_TAG}_{TAG}_{TS}.txt"
out_path = f"data/processed/all_experiment_results_{_MODEL_TAG}_{TAG}_{TS}.csv"
tee = Tee(log_path); sys.stdout = tee

print(f"[v2 main] log: {log_path}")
print(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"投票次数: 1")
print("=" * 60)

# Patch SECTOR_SENSITIVITY to v2 BEFORE invoking run_analysis
import main as _main
ORIG = dict(_main.SECTOR_SENSITIVITY)
_main.SECTOR_SENSITIVITY = V2_MATRIX
print(f"\n[v2 prior] {len(V2_MATRIX)} entries (vs v1: {len(ORIG)} entries)")
print(f"[v2 prior] continuous-valued, hierarchical shrinkage with n_full=10")
print()

car_df = pd.read_csv("data/processed/car_results.csv")
targets = (
    car_df[car_df["window"] == 5]
    [["event_date", "event_name", "industry_code", "direction", "CAR"]]
    .drop_duplicates().reset_index(drop=True)
)
print(f"\n共 {len(targets)} 个场景")

# ── 续跑支持：读取已有的同种子部分结果 CSV，跳过已成功场景 ──────
RESUME_FROM = os.environ.get("RESUME_FROM", "")
done_keys: set[tuple[str, str]] = set()
results: list[dict] = []
if RESUME_FROM and os.path.exists(RESUME_FROM):
    prev = pd.read_csv(RESUME_FROM)
    prev["event_date"]    = prev["event_date"].astype(str).str[:10]
    prev["industry_code"] = prev["industry_code"].astype(str)
    done_keys = set(zip(prev["event_date"], prev["industry_code"]))
    results = prev.to_dict("records")
    print(f"[续跑] 从 {RESUME_FROM} 载入 {len(done_keys)} 个已完成场景")
    print(f"[续跑] 待跑场景: {len(targets) - len(done_keys)}\n")
else:
    print()
for idx, row in targets.iterrows():
    event_date    = str(row["event_date"])[:10]
    industry_code = str(row["industry_code"])
    event_name    = row["event_name"]
    real_car      = float(row["CAR"])
    real_dir      = "+" if real_car >= 0 else "-"

    if (event_date, industry_code) in done_keys:
        continue

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

        # Disagreement-as-feature fields (added for paper §5.X)
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

    # 增量保存：每 10 个场景写一次，挂了不丢
    if (idx + 1) % 10 == 0 and len(results) > 0:
        pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  [增量保存] {len(results)} 条已落盘 → {out_path}")

res_df = pd.DataFrame(results)
res_df.to_csv(out_path, index=False, encoding="utf-8-sig")

print("\n" + "="*60)
print(f"v2 main 完成（hierarchical shrinkage prior）")
print("="*60)
if len(res_df) > 0:
    acc = res_df["dir_correct"].mean()
    print(f"\nCAR 符号准确率: {acc:.2%} ({res_df['dir_correct'].sum()}/{len(res_df)})")
    print(f"\n对比基准:")
    print(f"  v1 main (Stage 1+2 + 92 docs):       64.77%  ← 论文当前主结果")
    print(f"  v2 main (hierarchical shrinkage):    {acc:.2%}")
    delta = acc - 0.6477
    print(f"\n  vs v1 main 差值: {delta:+.2%}")
    if delta >= -0.02:
        print(f"  ✅ v2 几乎无退化或更好，可作新主结果")
    elif delta >= -0.05:
        print(f"  ⚠️  v2 略低于 v1，但 LOO 鲁棒性可能弥补")
    else:
        print(f"  ❌ v2 明显退化，需要调整 n_full / anchors 后重试")

    pos = res_df[res_df["real_dir"] == "+"]
    neg = res_df[res_df["real_dir"] == "-"]
    print(f"\n正向事件准确率: {pos['dir_correct'].mean():.2%} ({pos['dir_correct'].sum()}/{len(pos)})")
    print(f"负向事件准确率: {neg['dir_correct'].mean():.2%} ({neg['dir_correct'].sum()}/{len(neg)})")
    print(f"平均 MAE: {res_df['MAE'].mean():.6f}")

    # T+k
    print(f"\n关键时点累积方向:")
    for k in (1, 3, 5, 10):
        col = f"CAR_dir_T{k}"
        if col in res_df.columns:
            valid = res_df[res_df[col].notna()]
            if len(valid) > 0:
                a = valid[col].astype(bool).mean()
                print(f"  T+{k}: {a:.2%}")

    # Per-event-type
    car_t5 = car_df[car_df["window"] == 5][
        ["event_date", "industry_code", "event_type"]].copy()
    car_t5["event_date"] = car_t5["event_date"].astype(str).str[:10]
    car_t5["industry_code"] = car_t5["industry_code"].astype(str)
    res_df["event_date"] = res_df["event_date"].astype(str).str[:10]
    res_df["industry_code"] = res_df["industry_code"].astype(str)
    merged = res_df.merge(car_t5, on=["event_date", "industry_code"], how="left")
    if "event_type" in merged.columns:
        type_acc = merged.groupby("event_type")["dir_correct"].agg(["mean", "count"])
        print(f"\n各事件类型准确率:")
        print(type_acc)

print(f"\nCSV: {out_path}")
print(f"日志: {log_path}")

sys.stdout = tee.terminal
tee.close()
print(f"\n✅ 完成: {log_path}")
