"""
Analyze the v4 ablation matrix: is any config difference REAL or just noise?

Loads base/irf/fund/all walk-forward CSVs, aligns by (event_date, industry),
and reports for each config vs base:
  - overall / positive / negative accuracy with 95% bootstrap CIs
  - paired bootstrap p-value on the accuracy gap (overall & negative class)
  - MAE mean + paired difference (does IRF really cut MAE?)
  - per-event direction flips vs base: how many changed, and of those how many
    flipped correct→wrong vs wrong→correct

Motivation: single-seed n=168 has ~±2.4pp run-to-run noise (base was 61.90%
on 07-14 and 59.52% on 07-16 — same config). So ±1-3pp ablation gaps must be
significance-tested before being called an effect.

Run:  python scripts/analyze_ablation_matrix.py
      python scripts/analyze_ablation_matrix.py --nboot 5000
"""
from __future__ import annotations
import os, sys, glob, argparse
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

TAGS = ["base", "irf", "fund", "fundval", "all"]


def latest(tag):
    c = glob.glob(f"data/processed/all_experiment_results_qwen2.5_wf_{tag}_*.csv")
    return max(c, key=os.path.getmtime) if c else None


def key(d):
    return (d["event_date"].astype(str).str[:10] + "|" +
            d["industry_code"].astype(str))


def boot_ci(x, nboot, rng):
    x = np.asarray(x, float)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(nboot)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def paired_p(a, b, nboot, rng):
    """paired bootstrap two-sided p that mean(a)-mean(b) != 0 (a,b aligned 0/1)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    obs = d.mean()
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(nboot)])
    # two-sided p: fraction of bootstrap means on the opposite side of 0 from obs
    p = 2.0 * min((bs <= 0).mean(), (bs >= 0).mean()) if obs != 0 else 1.0
    return obs, min(p, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=5000)
    args = ap.parse_args()
    rng = np.random.default_rng(20260717)

    dfs = {}
    for t in TAGS:
        p = latest(t)
        if not p:
            print(f"{t}: MISSING"); continue
        d = pd.read_csv(p); d["k"] = key(d)
        dfs[t] = d.set_index("k")
        print(f"{t:<5} {os.path.basename(p)}  n={len(d)}")
    if "base" not in dfs:
        print("no base CSV"); return 1

    base = dfs["base"]
    print("\n" + "=" * 78)
    print(f"{'config':<6}{'overall (95%CI)':>26}{'neg (95%CI)':>26}{'MAE':>10}")
    print("=" * 78)
    for t in TAGS:
        if t not in dfs:
            continue
        d = dfs[t]
        acc = d["dir_correct"].astype(int)
        neg = d[d["real_dir"] == "-"]["dir_correct"].astype(int)
        lo, hi = boot_ci(acc, args.nboot, rng)
        nlo, nhi = boot_ci(neg, args.nboot, rng)
        print(f"{t:<6}{acc.mean():>7.2%} [{lo:.2%},{hi:.2%}]"
              f"{neg.mean():>10.2%} [{nlo:.2%},{nhi:.2%}]{d['MAE'].mean():>10.4f}")

    print("\n" + "=" * 78)
    print("vs base — paired bootstrap significance + direction flips")
    print("=" * 78)
    for t in ["irf", "fund", "fundval", "all"]:
        if t not in dfs:
            continue
        d = dfs[t]
        common = base.index.intersection(d.index)
        b = base.loc[common]; x = d.loc[common]
        # overall
        go, po = paired_p(x["dir_correct"].astype(int), b["dir_correct"].astype(int),
                          args.nboot, rng)
        # negative class
        negmask = b["real_dir"] == "-"
        gn, pn = paired_p(x.loc[negmask, "dir_correct"].astype(int),
                          b.loc[negmask, "dir_correct"].astype(int), args.nboot, rng)
        # MAE paired
        dmae = (x["MAE"] - b["MAE"])
        # direction flips
        flip = x["final_dir"] != b["final_dir"]
        n_flip = int(flip.sum())
        w2c = int(((~b["dir_correct"].astype(bool)) & x["dir_correct"].astype(bool) & flip).sum())
        c2w = int((b["dir_correct"].astype(bool) & (~x["dir_correct"].astype(bool)) & flip).sum())
        print(f"\n[{t} vs base]  (n_common={len(common)})")
        print(f"  overall Δ={go*100:+.2f}pp  p={po:.3f}  "
              f"{'SIGNIFICANT' if po<0.05 else 'not sig (noise)'}")
        print(f"  neg     Δ={gn*100:+.2f}pp  p={pn:.3f}  "
              f"{'SIGNIFICANT' if pn<0.05 else 'not sig (noise)'}")
        print(f"  MAE     Δ={dmae.mean():+.5f} (mean paired)  "
              f"{'improved' if dmae.mean()<0 else 'worse'}")
        print(f"  方向翻转 {n_flip} 个: 纠正(错→对) {w2c}, 帮倒忙(对→错) {c2w}, "
              f"净 {w2c - c2w:+d}")
    print("\n判读：单seed n=168 运行间噪声≈±2.4pp（base 07-14=61.90% vs 07-16=59.52%）。")
    print("p>0.05 的差异应视为噪声，不能当作组件有效/有害的证据。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
