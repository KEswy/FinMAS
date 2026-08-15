"""
Walk-forward honest baseline (v4 Phase 0)
=========================================

Runs the full pipeline over all events under the temporal firewall:
  - L1: sensitivity prior uses ONLY events with event_date < t (per-event,
        computed inside run_analysis via WalkForwardContext.sensitivity_prior).
  - L2: direction label is never read (walk_forward forces blind + no fallback).
  - L3/L4/L5: RAG retrieves only docs with pub_date < t, from the event-tied
        corpus (18 forward-looking mechanism docs quarantined).

This produces the HONEST walk-forward CAR-sign accuracy — expected to fall well
below the leaked full-data 61.90% (v3 LOO was 47.62%). Cold-start events (early
2020, no prior history) fall back to s=1.0 = LLM+TS only, which is correct.

Unlike run_all_events_v2.py, this does NOT patch a global SECTOR_SENSITIVITY:
each event builds its own leak-free prior. The env WALK_FORWARD is also set so
any downstream default respects the firewall.

Usage (PowerShell):
    $env:WALK_FORWARD="1"; $env:SEED="42"; python scripts/run_walk_forward.py

DeepSeek V4 Flash (recommended):
    $env:DEEPSEEK_API_KEY="sk-..."
    $env:LLM_MODEL="deepseek-v4-flash"
    $env:MAX_CONCURRENCY="4"
    $env:WALK_FORWARD="1"; $env:SEED="42"; python scripts/run_walk_forward.py
"""
from __future__ import annotations
import os, sys, datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

from agents.llm_client import get_llm_config

# force firewall on for this whole run
os.environ["WALK_FORWARD"] = "1"


class Tee:
    def __init__(self, p):
        self.terminal = sys.stdout
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        self.file = open(p, "w", encoding="utf-8")
    def write(self, m):
        with self.lock:
            try:
                self.terminal.write(m)
            except UnicodeEncodeError:
                self.terminal.write(m.encode("ascii", "replace").decode("ascii"))
            self.file.write(m)
            self.file.flush()
    def flush(self):
        with self.lock:
            self.terminal.flush()
            self.file.flush()
    def isatty(self):
        return self.terminal.isatty()
    def close(self):
        self.file.close()


_SEED_TAG = os.environ.get("SEED", "42")
_LLM_CONFIG = get_llm_config()
_LLM_PROVIDER = _LLM_CONFIG["provider"]
_MODEL_RAW = _LLM_CONFIG["model"]
_MODEL_TAG = _MODEL_RAW.split(":")[0].replace("/", "_")
_DEFAULT_WORKERS = 4 if _LLM_PROVIDER == "deepseek" else 1
MAX_WORKERS = max(1, int(os.environ.get("MAX_CONCURRENCY", str(_DEFAULT_WORKERS))))
# RUN_TAG 区分消融配置（如 base / irf / fund / all），避免四跑 CSV 只靠时间戳
_RUN_TAG = os.environ.get("RUN_TAG", "")
_seed_part = f"_seed{_SEED_TAG}" if _SEED_TAG != "42" else ""
TAG = f"wf{('_' + _RUN_TAG) if _RUN_TAG else ''}{_seed_part}"
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"data/results/experiment_{_MODEL_TAG}_{TAG}_{TS}.txt"
out_path = f"data/processed/all_experiment_results_{_MODEL_TAG}_{TAG}_{TS}.csv"
tee = Tee(log_path); sys.stdout = tee

print(f"[walk-forward] log: {log_path}")
print(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"模型: {_MODEL_RAW}  提供方: {_LLM_PROVIDER}  并发: {MAX_WORKERS}  SEED={_SEED_TAG}")
print(f"WALK_FORWARD=1  (L1 时间隔离先验 / L2 无标签 / L3-5 RAG pub_date<t)")
print("=" * 60)

import main as _main

_CAR_CSV = os.environ.get("CAR_CSV", "data/processed/car_results.csv")
print(f"[walk-forward] target benchmark: {_CAR_CSV}")
car_df = pd.read_csv(_CAR_CSV)
targets = (
    car_df[car_df["window"] == 5]
    [["event_date", "event_name", "industry_code", "direction", "CAR"]]
    .drop_duplicates().reset_index(drop=True)
)
print(f"\n共 {len(targets)} 个场景\n")

# resume support
RESUME_FROM = os.environ.get("RESUME_FROM", "")
done_keys: set = set()
results: list = []
if RESUME_FROM and os.path.exists(RESUME_FROM):
    prev = pd.read_csv(RESUME_FROM)
    prev["event_date"] = prev["event_date"].astype(str).str[:10]
    prev["industry_code"] = prev["industry_code"].astype(str)
    done_keys = set(zip(prev["event_date"], prev["industry_code"]))
    results = prev.to_dict("records")
    print(f"[续跑] 载入 {len(done_keys)} 个已完成场景，待跑 {len(targets)-len(done_keys)}\n")

success = failed = 0
results_lock = threading.Lock()


def _save_results():
    if not results:
        return
    with results_lock:
        frame = pd.DataFrame(results)
        if "_order" in frame.columns:
            frame = frame.sort_values("_order").drop(columns=["_order"])
        frame.to_csv(out_path, index=False, encoding="utf-8-sig")


def _process_one(task):
    idx, row = task
    event_date = str(row["event_date"])[:10]
    industry_code = str(row["industry_code"])
    key = (event_date, industry_code)
    if key in done_keys:
        return None

    event_name = row["event_name"]
    real_car = float(row["CAR"])
    real_dir = "+" if real_car >= 0 else "-"
    print(f"\n[{idx+1}/{len(targets)}] {event_date} | {industry_code} | {event_name[:40]}")

    try:
        r = _main.run_analysis(
            event=event_name,
            entity_names=[event_name[:20], industry_code],
            task_type="policy_event",
            industry_code=industry_code,
            event_date=event_date,
            walk_forward=True,      # ← 触发时间防火墙（L1/L2/L3-5）
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
                      for k in (1, 3, 5, 10) if f"CAR_dir_correct_T{k}" in kps}
        dis = r.get("disagreement", {}) or {}
        rk = r.get("risk", {}) or {}     # Phase 4 事件级风险
        rk_iv = rk.get("risk_interval", [0.0, 0.0])
        risk_by_horizon = r.get("risk_by_horizon", {}) or {}
        horizon_risk_fields = {}
        for _k in (1, 5, 20):
            _rk = risk_by_horizon.get(str(_k), {}) or {}
            _iv = _rk.get("risk_interval", [float("nan"), float("nan")])
            horizon_risk_fields.update({
                f"event_VaR_T{_k}": _rk.get("event_VaR", float("nan")),
                f"expected_MDD_T{_k}": _rk.get("expected_MDD", float("nan")),
                f"tail_prob_T{_k}": _rk.get("tail_prob", float("nan")),
                f"risk_lo_T{_k}": _iv[0],
                f"risk_hi_T{_k}": _iv[1],
            })

        record = {
            "event_date": event_date, "event_name": event_name[:35],
            "industry_code": industry_code, "real_CAR": real_car,
            "real_dir": real_dir, "final_dir": pred_dir, "dir_correct": correct,
            "MAE": mae,
            "RMSE": r["metrics"]["prediction"].get("RMSE", 0.0),
            "DirectionAcc": r["metrics"]["prediction"].get("DirectionAcc", 0.0),
            "Coverage_95": r["metrics"]["prediction"].get("Coverage_95", 0.0),
            "Sharpe": r["metrics"]["backtest"].get("SharpeRatio", 0.0),
            "WinRate": r["metrics"]["backtest"].get("WinRate", 0.0),
            "triggered": r.get("triggered_correction", False),
            "ts_pre_signal": dis.get("ts_pre_signal", ""),
            "llm_signal": dis.get("llm_signal", ""),
            "disagreement": dis.get("disagreement", False),
            "debate_abstained": dis.get("debate_abstained", None),
            "debate_confidence": dis.get("debate_confidence", None),
            "reld_r_ind": dis.get("reld_r_ind", None),
            "reld_r_mkt": dis.get("reld_r_mkt", None),
            "reld_beats_market": dis.get("reld_beats_market", None),
            "event_VaR": rk.get("event_VaR", float("nan")),
            "expected_MDD": rk.get("expected_MDD", float("nan")),
            "tail_prob": rk.get("tail_prob", float("nan")),
            "risk_lo": rk_iv[0], "risk_hi": rk_iv[1],
            "dispersion": rk.get("dispersion", float("nan")),
            **horizon_risk_fields,
            **kp_results,
            "_order": int(idx),
        }
        return {"ok": True, "record": record}
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return {"ok": False, "error": str(e)}


tasks = [(idx, row) for idx, row in targets.iterrows()]
if MAX_WORKERS == 1:
    for task in tasks:
        res = _process_one(task)
        if res is None:
            continue
        if res["ok"]:
            results.append(res["record"])
            success += 1
        else:
            failed += 1
        _save_results()
else:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_process_one, task) for task in tasks]
        for future in as_completed(futures):
            res = future.result()
            if res is None:
                continue
            if res["ok"]:
                results.append(res["record"])
                success += 1
            else:
                failed += 1
            _save_results()

# ── summary ─────────────────────────────────────────────
df = pd.DataFrame(results)
if "_order" in df.columns:
    df = df.sort_values("_order").drop(columns=["_order"])
_save_results()
print("\n" + "=" * 60)
print(f"walk-forward 完成：成功 {success} 失败 {failed} 总计 {len(targets)}")
if len(df) > 0:
    acc = df["dir_correct"].mean()
    n = len(df)
    print(f"\n{'─'*40}")
    print(f"【诚实 walk-forward CAR 符号准确率】")
    print(f"{'─'*40}")
    print(f"总准确率: {acc:.2%} ({int(df['dir_correct'].sum())}/{n})")
    pos = df[df["real_dir"] == "+"]["dir_correct"]
    neg = df[df["real_dir"] == "-"]["dir_correct"]
    if len(pos): print(f"正向事件: {pos.mean():.2%} ({int(pos.sum())}/{len(pos)})")
    if len(neg): print(f"负向事件: {neg.mean():.2%} ({int(neg.sum())}/{len(neg)})")
    print(f"平均 MAE: {df['MAE'].mean():.4f}")
    print(f"\n对比参考：v3 full-data(泄漏) 61.90% / v3 LOO 47.62%")
    print(f"本次 walk-forward: {acc:.2%}  (泄漏 delta vs full-data: {(acc-0.6190)*100:+.2f}pp)")

print(f"\nCSV: {out_path}")
print(f"日志: {log_path}")
print(f"结束: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
sys.stdout = tee.terminal
tee.close()
print(f"✅ walk-forward 完成，日志: {log_path}")
