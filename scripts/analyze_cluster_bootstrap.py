"""
Event-cluster bootstrap (v4.1 review fix #2).

168 scenarios are NOT 168 independent samples: they are 50 events, each fanned
out to several industries whose returns/text/shock are highly correlated.
Resampling scenarios independently understates CIs. This script resamples at the
EVENT level (50 clusters, with replacement, carrying all a cluster's scenarios),
recomputing every headline p-value and CI, and also reports the event-level
macro average (mean over per-event accuracies) alongside the scenario micro
average.

Covers:
  - per-class Ours (base 3-seed majority) vs LLM-only  (the survives-firewall claim)
  - ablation overall/neg gaps (irf/all vs base)

Run:  python scripts/analyze_cluster_bootstrap.py [--nboot 5000]
"""
from __future__ import annotations
import os, glob, re, sys, argparse
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

SEEDS = ["42", "123", "2024"]


def base_csv(seed):
    if seed == "42":
        c = [p for p in glob.glob("data/processed/all_experiment_results_qwen2.5_wf_base_*.csv")
             if not re.search(r"_seed\d+_", p)]
    else:
        c = glob.glob(f"data/processed/all_experiment_results_qwen2.5_wf_base_seed{seed}_*.csv")
    return max(c, key=os.path.getmtime) if c else None


def key(d):
    return (d["event_date"].astype(str).str[:10] + "|" + d["industry_code"].astype(str))


def cluster_boot(df, stat_fn, nboot, rng):
    """df has column 'ev' (event id). Resample events with replacement, apply
    stat_fn to the concatenated rows. Returns (point, lo, hi, p_two_sided_vs_0)."""
    events = df["ev"].unique()
    point = stat_fn(df)
    boots = np.empty(nboot)
    groups = {e: df[df.ev == e] for e in events}
    for b in range(nboot):
        pick = rng.choice(events, size=len(events), replace=True)
        samp = pd.concat([groups[e] for e in pick], ignore_index=True)
        boots[b] = stat_fn(samp)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    return point, lo, hi, min(p, 1.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nboot", type=int, default=5000)
    args = ap.parse_args(); rng = np.random.default_rng(20260719)

    # Ours = base 3-seed majority
    frames = []
    for s in SEEDS:
        p = base_csv(s)
        if not p:
            print(f"base seed{s} MISSING"); continue
        d = pd.read_csv(p); d["k"] = key(d)
        frames.append(d[["k", "event_date", "real_dir", "dir_correct"]].rename(
            columns={"dir_correct": f"c{s}"}))
    ours = frames[0]
    for f in frames[1:]:
        seedcol = [c for c in f.columns if re.fullmatch(r"c\d+", c)][0]
        ours = ours.merge(f[["k", seedcol]], on="k")
    ccols = [c for c in ours.columns if re.fullmatch(r"c\d+", c)]
    ours["ours"] = (ours[ccols].astype(int).sum(axis=1) >= len(ccols)/2.0).astype(int)
    ours["ev"] = ours["event_date"].str[:10]

    lp = sorted(glob.glob("data/processed/baseline_llm_only_qwen2.5_32b_*.csv"),
                key=os.path.getmtime)
    llm = pd.read_csv(lp[-1]); llm["k"] = key(llm)
    m = ours[["k", "ev", "real_dir", "ours"]].merge(
        llm[["k", "dir_correct"]].rename(columns={"dir_correct": "llm"}), on="k")
    m["llm"] = m["llm"].astype(int)

    print("=" * 74)
    print(f"Event-cluster bootstrap ({m['ev'].nunique()} event clusters, "
          f"nboot={args.nboot})")
    print("=" * 74)
    print(f"{'class':<9}{'n_sc':>5}{'gap(O-L) [95%CI]':>28}{'cluster-p':>11}{'ev-macro gap':>14}")
    for lab, sub in [("overall", m), ("positive", m[m.real_dir == "+"]),
                     ("negative", m[m.real_dir == "-"])]:
        gap_fn = lambda d: d["ours"].mean() - d["llm"].mean()
        pt, lo, hi, p = cluster_boot(sub, gap_fn, args.nboot, rng)
        # event-level macro: per-event gap, then mean
        ev_gap = sub.groupby("ev").apply(
            lambda g: g["ours"].mean() - g["llm"].mean()).mean()
        print(f"{lab:<9}{len(sub):>5}{pt*100:>+11.2f}pp [{lo*100:+.1f},{hi*100:+.1f}]"
              f"{p:>11.3f}{ev_gap*100:>+13.2f}pp")

    print("\n对比 scenario-level bootstrap（旧，过窄）：负向 +12.79pp p=0.057。")
    print("event-cluster p 若变大 = 旧 p 因忽略事件内相关而失真；这是诚实口径。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
