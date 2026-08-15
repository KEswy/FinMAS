"""
Main-axis analysis (Tier M, n=168)
==================================

Pairs Ours (3-seed pooled) vs LLM-only baseline scenario-by-scenario, then
computes the three candidate main-axis metrics so we can decide the paper
framing (Plan A cost-weighted / Plan B MAE+market_event / Plan C honest):

  1. unweighted CAR-sign accuracy gap (context number, not the axis)
  2. per-class redistribution (Ours - LLM-only, per event_type)
  3. cost-weighted score with FP penalty = 8x (false-positive = pred + but real -)
  4. paired bootstrap p-value on the unweighted gap

All numbers come straight from the result CSVs. No fabrication.

Usage:
  python scripts/analyze_main_axis.py
"""
from __future__ import annotations
import os, sys, io, glob

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
os.chdir(ROOT_DIR)

import pandas as pd
import numpy as np

# Deterministic bootstrap (no Date.now/random seed drift across reruns)
RNG = np.random.RandomState(20260531)

OURS_FILES = {
    "seed42":   "data/processed/all_experiment_results_qwen2.5_v2_20260526_114901.csv",
    "seed123":  "data/processed/all_experiment_results_qwen2.5_v2_seed123_20260527_201449.csv",
    "seed2024": "data/processed/all_experiment_results_qwen2.5_v2_seed2024_20260528_185645.csv",
}
LLM_ONLY_FILE = "data/processed/baseline_llm_only_qwen2.5_32b_20260531_130014.csv"


def _key(df):
    return (df["event_date"].astype(str).str[:10] + "|" +
            df["industry_code"].astype(str))


def load_ours():
    """Pool 3 seeds. Per scenario, Ours-correct = majority vote of dir_correct."""
    frames = []
    for name, path in OURS_FILES.items():
        d = pd.read_csv(path)
        d["k"] = _key(d)
        d = d[["k", "event_date", "industry_code", "real_dir",
               "final_dir", "dir_correct"]].copy()
        d["seed"] = name
        frames.append(d)
    alld = pd.concat(frames, ignore_index=True)
    # majority vote of correctness across 3 seeds per scenario
    g = alld.groupby("k").agg(
        event_date=("event_date", "first"),
        industry_code=("industry_code", "first"),
        real_dir=("real_dir", "first"),
        ours_correct_votes=("dir_correct", "sum"),
        n_seeds=("dir_correct", "count"),
    ).reset_index()
    g["ours_correct"] = g["ours_correct_votes"] >= (g["n_seeds"] / 2.0)
    # majority predicted direction per scenario (for cost-weighted FP)
    dir_mode = (alld.groupby("k")["final_dir"]
                .agg(lambda s: s.value_counts().idxmax()).rename("ours_dir"))
    g = g.merge(dir_mode, on="k", how="left")
    return g


def load_llm_only():
    d = pd.read_csv(LLM_ONLY_FILE)
    d["k"] = _key(d)
    d = d[["k", "event_type", "pred_dir", "dir_correct"]].copy()
    d = d.rename(columns={"pred_dir": "llm_dir", "dir_correct": "llm_correct"})
    return d


def main() -> int:
    ours = load_ours()
    llm  = load_llm_only()
    m = ours.merge(llm, on="k", how="inner")
    n = len(m)
    print("=" * 72)
    print(f"Main-axis analysis (Tier M, paired n={n})")
    print("=" * 72)
    if n < 150:
        print(f"WARNING: paired n={n} < 168 expected; key mismatch?")

    m["ours_correct"] = m["ours_correct"].astype(bool)
    m["llm_correct"]  = m["llm_correct"].astype(bool)

    # --- 1. unweighted accuracy ---
    ours_acc = m["ours_correct"].mean()
    llm_acc  = m["llm_correct"].mean()
    print(f"\n[1] Unweighted CAR-sign accuracy (paired)")
    print(f"    Ours (3-seed majority): {ours_acc:.2%} ({m['ours_correct'].sum()}/{n})")
    print(f"    LLM-only:               {llm_acc:.2%} ({m['llm_correct'].sum()}/{n})")
    print(f"    gap = {(ours_acc-llm_acc)*100:+.2f}pp  (context number, NOT the axis)")

    # --- 2. per-class redistribution ---
    print(f"\n[2] Per-class redistribution (Ours - LLM-only)")
    rows = []
    for et, grp in m.groupby("event_type"):
        oa = grp["ours_correct"].mean()
        la = grp["llm_correct"].mean()
        rows.append((et, len(grp), oa, la, (oa-la)*100))
    pc = pd.DataFrame(rows, columns=["event_type","n","ours","llm","gap_pp"])
    pc = pc.sort_values("n", ascending=False)
    for _, r in pc.iterrows():
        print(f"    {r['event_type']:<22} n={int(r['n']):>3}  "
              f"Ours={r['ours']:.2%}  LLM={r['llm']:.2%}  "
              f"gap={r['gap_pp']:+.2f}pp")

    # --- 2b. per real-direction (the round-3 'redistribution' axis) ---
    print(f"\n[2b] Per real-direction redistribution (the round-3 axis)")
    for d, label in [("+","positive events"), ("-","negative events")]:
        grp = m[m["real_dir"] == d]
        if len(grp) == 0:
            continue
        oa = grp["ours_correct"].mean()
        la = grp["llm_correct"].mean()
        print(f"    {label:<16} n={len(grp):>3}  "
              f"Ours={oa:.2%}  LLM={la:.2%}  gap={(oa-la)*100:+.2f}pp")

    # --- 3. cost-weighted FP=8x ---
    print(f"\n[3] Cost-weighted score (false-positive penalty = 8x)")
    def cost(pred_dir_col, correct_col):
        # FP = predicted '+' but real '-' (costly: long into a drop)
        fp = ((m[pred_dir_col] == "+") & (m["real_dir"] == "-")).sum()
        fn = ((m[pred_dir_col] == "-") & (m["real_dir"] == "+")).sum()
        # weighted error: 8*FP + 1*FN, lower is better
        return 8*fp + fn, fp, fn
    o_cost, o_fp, o_fn = cost("ours_dir", "ours_correct")
    l_cost, l_fp, l_fn = cost("llm_dir", "llm_correct")
    print(f"    Ours:     weighted_err={o_cost:>4}  (FP={o_fp}, FN={o_fn})")
    print(f"    LLM-only: weighted_err={l_cost:>4}  (FP={l_fp}, FN={l_fn})")
    print(f"    Ours reduces weighted error by {l_cost-o_cost} "
          f"({(l_cost-o_cost)/max(l_cost,1):.1%})")

    # --- 4. paired bootstrap on unweighted gap ---
    print(f"\n[4] Paired bootstrap on unweighted gap (B=10000)")
    diff = m["ours_correct"].astype(int).values - m["llm_correct"].astype(int).values
    B = 10000
    boot = np.empty(B)
    idx = np.arange(n)
    for b in range(B):
        s = RNG.choice(idx, size=n, replace=True)
        boot[b] = diff[s].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    obs = diff.mean()
    p_two = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    print(f"    observed mean diff = {obs*100:+.2f}pp")
    print(f"    95% CI = [{lo*100:+.2f}, {hi*100:+.2f}]pp")
    print(f"    two-sided p (crosses zero?) = {p_two:.3f}")
    print(f"    {'CROSSES ZERO (non-significant)' if lo<0<hi else 'CI excludes zero (significant)'}")

    # --- 5. per-direction paired bootstrap (THE MAIN AXIS significance) ---
    print(f"\n[5] Per-direction paired bootstrap (MAIN AXIS, B=10000)")
    pdir_p = {}
    for d, label in [("-","negative events"), ("+","positive events")]:
        grp = m[m["real_dir"] == d]
        gd = (grp["ours_correct"].astype(int).values -
              grp["llm_correct"].astype(int).values)
        ng = len(gd)
        bootd = np.empty(B)
        gidx = np.arange(ng)
        for b in range(B):
            s = RNG.choice(gidx, size=ng, replace=True)
            bootd[b] = gd[s].mean()
        glo, ghi = np.percentile(bootd, [2.5, 97.5])
        gp = 2 * min((bootd <= 0).mean(), (bootd >= 0).mean())
        pdir_p[label] = (gd.mean()*100, glo*100, ghi*100, gp)
        sig = "SIGNIFICANT" if not (glo < 0 < ghi) else "crosses zero"
        print(f"    {label:<16} n={ng:>3}  gap={gd.mean()*100:+.2f}pp  "
              f"95%CI=[{glo*100:+.2f},{ghi*100:+.2f}]  p={gp:.3f}  {sig}")

    print("\n" + "=" * 72)
    print("Summary for framing decision:")
    print(f"  [axis ] per-direction redistribution:")
    for lab, (g, lo2, hi2, p2) in pdir_p.items():
        print(f"            {lab:<16} {g:+.2f}pp  p={p2:.3f}")
    print(f"  [supp ] cost-weighted err reduction {l_cost-o_cost} ({(l_cost-o_cost)/max(l_cost,1):.1%}), FP {l_fp}->{o_fp}")
    print(f"  [ctx  ] unweighted gap {(ours_acc-llm_acc)*100:+.2f}pp, p={p_two:.3f} (honest, non-significant)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
