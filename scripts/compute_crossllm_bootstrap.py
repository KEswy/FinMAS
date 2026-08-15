"""
Cross-LLM Bootstrap CI on the 88-event benchmark.

Loads three corrected-training 88-event runs (Qwen-32B / Llama-8B /
DeepSeek-R1-32B, all seed=42 unless noted), computes:

  1. Per-LLM CAR sign accuracy + 95% bootstrap CI (n=2000 resamples,
     resample events with replacement, take mean)
  2. Pairwise final_dir agreement (X / 88) and mean |dMAE| per pair —
     this measures whether different LLMs make the same SIGN decisions
     while possibly differing in MAGNITUDE
  3. Pairwise CAR accuracy gap + 95% bootstrap CI (paired resampling:
     resample event indices, compute gap on same indices, repeat)
  4. Per-event-type accuracy for each LLM

Output: console table pasteable into LaTeX, plus a CSV summary.

Usage:
    python scripts/compute_crossllm_bootstrap.py
    python scripts/compute_crossllm_bootstrap.py --n_bootstrap 5000
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
os.chdir(ROOT_DIR)

# Tier M n=168 runs (seed=42, blind), all read the same 168-scenario car_results.csv.
# Qwen = seed42 main run; Llama / DeepSeek = 2026-07-13 cross-LLM reruns.
LLM_FILES = {
    "Qwen-2.5-32B":    "data/processed/all_experiment_results_qwen2.5_v2_20260526_114901.csv",
    "Llama-3.1-8B":    "data/processed/all_experiment_results_llama3.1_v2_20260713_192056.csv",
    "DeepSeek-R1-32B": "data/processed/all_experiment_results_deepseek-r1_v2_20260713_203514.csv",
}

CAR_CSV = "data/processed/car_results.csv"


def load_and_align(files: dict) -> dict:
    car = pd.read_csv(CAR_CSV)
    car_t5 = car[car["window"] == 5][["event_date", "industry_code", "event_type"]].copy()
    car_t5["event_date"] = car_t5["event_date"].astype(str).str[:10]
    car_t5["industry_code"] = car_t5["industry_code"].astype(str)
    car_t5["key"] = car_t5["event_date"] + "_" + car_t5["industry_code"]

    dfs = {}
    for name, path in files.items():
        df = pd.read_csv(path)
        df["event_date"] = df["event_date"].astype(str).str[:10]
        df["industry_code"] = df["industry_code"].astype(str)
        df["key"] = df["event_date"] + "_" + df["industry_code"]
        df = df.merge(car_t5[["key", "event_type"]], on="key", how="left")
        df = df.sort_values("key").reset_index(drop=True)
        dfs[name] = df
        print(f"  {name:<20}  n={len(df)}  CAR={df['dir_correct'].mean()*100:.2f}%  "
              f"MAE={df['MAE'].mean():.4f}  file={os.path.basename(path)}")
    # sanity check: all use same event order
    keys_qwen = dfs["Qwen-2.5-32B"]["key"].tolist()
    for name in dfs:
        if dfs[name]["key"].tolist() != keys_qwen:
            print(f"WARN: {name} event order differs from Qwen")
    return dfs


def bootstrap_ci(correct: np.ndarray, n_boot: int, seed: int = 0):
    """resample-with-replacement bootstrap CI for a binary accuracy."""
    n = len(correct)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = correct[idx].mean()
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_gap_bootstrap_ci(c_a: np.ndarray, c_b: np.ndarray,
                              n_boot: int, seed: int = 0):
    """paired bootstrap CI for accuracy gap (acc_a - acc_b)."""
    n = len(c_a)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = c_a[idx].mean() - c_b[idx].mean()
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260516)
    p.add_argument("--out", default="data/processed/crossllm_bootstrap_summary.csv")
    args = p.parse_args()

    print("=" * 78)
    print(f"Cross-LLM Bootstrap CI on 168-scenario benchmark "
          f"(n_boot={args.n_bootstrap}, seed={args.seed})")
    print("=" * 78)
    print("\nLoading 3 LLM CSVs...")
    dfs = load_and_align(LLM_FILES)

    # ── 1. Per-LLM accuracy + 95% CI ─────────────────────────
    print("\n" + "=" * 78)
    print("1. Per-LLM CAR sign accuracy + 95% bootstrap CI")
    print("=" * 78)
    rows = []
    for name, df in dfs.items():
        correct = df["dir_correct"].astype(int).values
        acc = correct.mean()
        lo, hi = bootstrap_ci(correct, args.n_bootstrap, args.seed)
        mae = df["MAE"].mean()
        sharpe = df["Sharpe"].mean() if "Sharpe" in df.columns else np.nan
        rows.append({"llm": name, "acc": acc, "ci_lo": lo, "ci_hi": hi,
                       "mae": mae, "sharpe": sharpe})
        print(f"  {name:<20}  {acc*100:5.2f}%   "
              f"[95% CI {lo*100:.2f}, {hi*100:.2f}]   "
              f"MAE={mae:.4f}   Sharpe={sharpe:+.3f}")

    # ── 2. Pairwise final_dir agreement + MAE difference ────
    print("\n" + "=" * 78)
    print("2. Pairwise final_dir agreement (sign-level) & magnitude difference")
    print("=" * 78)
    pairs = [("Qwen-2.5-32B", "Llama-3.1-8B"),
             ("Qwen-2.5-32B", "DeepSeek-R1-32B"),
             ("Llama-3.1-8B", "DeepSeek-R1-32B")]
    pair_rows = []
    for a, b in pairs:
        da, db = dfs[a], dfs[b]
        m = da.merge(db, on="key", suffixes=("_a", "_b"))
        agree_dir = (m["final_dir_a"] == m["final_dir_b"]).sum()
        agree_corr = (m["dir_correct_a"] == m["dir_correct_b"]).sum()
        mae_diff = (m["MAE_a"] - m["MAE_b"]).abs().mean()
        pair_rows.append({"pair": f"{a} vs {b}",
                          "final_dir_agree": int(agree_dir),
                          "n": len(m),
                          "mae_abs_diff": mae_diff})
        print(f"  {a:<16} vs {b:<16}   "
              f"final_dir agree = {agree_dir}/{len(m)}   "
              f"mean |dMAE| = {mae_diff:.4f}")

    # ── 3. Pairwise accuracy gap + 95% bootstrap CI ─────────
    print("\n" + "=" * 78)
    print("3. Pairwise CAR sign accuracy gap (a - b) + 95% paired bootstrap CI")
    print("=" * 78)
    for a, b in pairs:
        da, db = dfs[a], dfs[b]
        # aligned by 'key'
        m = da[["key", "dir_correct"]].merge(
            db[["key", "dir_correct"]], on="key", suffixes=("_a", "_b"))
        c_a = m["dir_correct_a"].astype(int).values
        c_b = m["dir_correct_b"].astype(int).values
        gap = c_a.mean() - c_b.mean()
        lo, hi = paired_gap_bootstrap_ci(c_a, c_b, args.n_bootstrap, args.seed + hash(a+b)%1000)
        marker = ""
        if lo * hi > 0:
            marker = " ★ excludes 0 (significant)"
        print(f"  {a:<16} - {b:<16}   "
              f"Δacc = {gap*100:+5.2f}pp   "
              f"[95% CI {lo*100:+.2f}, {hi*100:+.2f}]{marker}")
        pair_rows[-1] if False else None  # ignore; pair rows already complete

    # ── 4. Per-event-type accuracy per LLM ──────────────────
    print("\n" + "=" * 78)
    print("4. Per-event-type accuracy per LLM")
    print("=" * 78)
    types = sorted(dfs["Qwen-2.5-32B"]["event_type"].dropna().unique())
    header = f"  {'event_type':<24}  {'n':>3}  " + "  ".join(
        f"{name:<16}" for name in dfs.keys())
    print(header)
    print("  " + "-" * (len(header) - 2))
    for et in types:
        line = f"  {et:<24}  "
        ns = []
        for name, df in dfs.items():
            sub = df[df["event_type"] == et]
            ns.append(len(sub))
            acc = sub["dir_correct"].mean() * 100
            line += f"  {acc:>13.2f}%"
        n = ns[0]  # all LLMs should have same per-type n
        print(f"  {et:<24}  {n:>3}  " + line.split(et)[1].strip())

    # ── Save summary CSV ────────────────────────────────────
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nPer-LLM summary saved: {args.out}")

    # ── 5. LaTeX-ready key numbers ──────────────────────────
    print("\n" + "=" * 78)
    print("5. LaTeX-paste-ready summary (for paper §5.9)")
    print("=" * 78)
    print()
    for row in rows:
        print(f"  {row['llm']:<20}: {row['acc']*100:.2f}\\% "
              f"[{row['ci_lo']*100:.2f}, {row['ci_hi']*100:.2f}]")
    print()
    print("Key narrative numbers:")
    qwen_r1_agree = next((r["final_dir_agree"] for r in pair_rows
                            if "DeepSeek" in r["pair"] and "Qwen" in r["pair"]), None)
    qwen_llama_agree = next((r["final_dir_agree"] for r in pair_rows
                              if "Llama" in r["pair"] and "Qwen" in r["pair"]), None)
    n_total = len(dfs["Qwen-2.5-32B"])
    print(f"  Qwen vs R1   sign agreement: {qwen_r1_agree}/{n_total}  "
          f"({'IDENTICAL' if qwen_r1_agree == n_total else 'differs'})")
    print(f"  Qwen vs Llama sign agreement: {qwen_llama_agree}/{n_total}")


if __name__ == "__main__":
    main()
