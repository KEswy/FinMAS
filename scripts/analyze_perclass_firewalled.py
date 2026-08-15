"""
Firewalled per-class redistribution: does v3's headline (Ours moves accuracy
from the positive class the LLM over-commits to, toward the negative class)
survive under the temporal firewall?

Ours = base config (walk-forward, firewalled), 3-seed per-event majority vote.
LLM-only = direct-prompt baseline (baseline_llm_only_qwen2.5_32b_*.csv). The
LLM-only baseline is ALREADY firewall-clean: it reads only the event
description text — no sensitivity prior, no RAG, no direction label
(see baseline_llm_only.py). So this comparison is leakage-free on both sides.

Reports per-direction accuracy for Ours and LLM-only, the gap (Ours − LLM),
and a paired-bootstrap p on each class gap. Compares to v3's leaky full-data
finding (neg +18.60pp p=0.002 / pos −14.63pp p=0.008).

Run:  python scripts/analyze_perclass_firewalled.py
"""
from __future__ import annotations
import os, glob, re, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

SEEDS = ["42", "123", "2024"]
NBOOT = 5000


def base_csv(seed):
    if seed == "42":
        c = [p for p in glob.glob("data/processed/all_experiment_results_qwen2.5_wf_base_*.csv")
             if not re.search(r"_seed\d+_", p)]
    else:
        c = glob.glob(f"data/processed/all_experiment_results_qwen2.5_wf_base_seed{seed}_*.csv")
    return max(c, key=os.path.getmtime) if c else None


def key(d):
    return (d["event_date"].astype(str).str[:10] + "|" + d["industry_code"].astype(str))


def paired_p(a, b, rng):
    d = np.asarray(a, float) - np.asarray(b, float)
    obs = d.mean()
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(NBOOT)])
    p = 2.0 * min((bs <= 0).mean(), (bs >= 0).mean()) if obs != 0 else 1.0
    return obs, min(p, 1.0)


def main():
    rng = np.random.default_rng(20260719)
    # Ours: 3-seed majority vote per event
    frames = []
    for s in SEEDS:
        p = base_csv(s)
        if not p:
            print(f"base seed{s}: MISSING"); continue
        d = pd.read_csv(p); d["k"] = key(d)
        frames.append(d[["k", "real_dir", "dir_correct"]].rename(
            columns={"dir_correct": f"c{s}"}))
        print(f"Ours base seed{s}: {os.path.basename(p)}")
    ours = frames[0]
    for f in frames[1:]:
        ours = ours.merge(f.drop(columns="real_dir"), on="k")
    ccols = [c for c in ours.columns if c.startswith("c")]
    ours["ours_correct"] = (ours[ccols].astype(int).sum(axis=1) >= len(ccols)/2.0).astype(int)

    # LLM-only baseline (already firewall-clean)
    lp = sorted(glob.glob("data/processed/baseline_llm_only_qwen2.5_32b_*.csv"),
                key=os.path.getmtime)
    if not lp:
        print("LLM-only baseline CSV MISSING"); return 1
    llm = pd.read_csv(lp[-1]); llm["k"] = key(llm)
    print(f"LLM-only: {os.path.basename(lp[-1])}\n")
    m = ours[["k", "real_dir", "ours_correct"]].merge(
        llm[["k", "dir_correct"]].rename(columns={"dir_correct": "llm_correct"}), on="k")
    m["llm_correct"] = m["llm_correct"].astype(int)

    print("=" * 68)
    print("Firewalled per-class: Ours (base 3-seed maj) vs LLM-only")
    print("=" * 68)
    print(f"{'class':<10}{'n':>5}{'Ours':>9}{'LLM-only':>10}{'gap(O-L)':>11}{'p':>8}")
    for lab, dsel in [("overall", m), ("positive", m[m.real_dir == "+"]),
                      ("negative", m[m.real_dir == "-"])]:
        o = dsel["ours_correct"].mean()
        l = dsel["llm_correct"].mean()
        gap, p = paired_p(dsel["ours_correct"], dsel["llm_correct"], rng)
        print(f"{lab:<10}{len(dsel):>5}{o*100:>8.2f}%{l*100:>9.2f}%"
              f"{gap*100:>+10.2f}pp{p:>8.3f}")

    print("\n对比 v3 全档案(泄漏)：负向 gap +18.60pp p=0.002 / 正向 −14.63pp p=0.008。")
    print("防火墙下若 gap 收缩/不显著 → v3 招牌 redistribution 部分依赖泄漏；")
    print("若仍显著同向 → redistribution 是真实的 model-vs-baseline 效应，独立于泄漏。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
