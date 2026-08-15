"""
Multi-seed aggregation (v4 noise reduction).

For base/irf/all across seeds 42/123/2024, reports the direction metric as a
3-seed mean±std AND a per-event majority vote (2-of-3), plus 3-seed mean MAE.
This turns the single-seed point estimates (swamped by ±2.4pp run noise) into
CI'd numbers, and checks whether IRF's MAE gain and any direction effect
survive averaging.

File matching is EXACT per seed to avoid the all/fund ambiguity:
  seed 42   -> ..._wf_<cfg>_<ts>.csv        (no _seed suffix)
  seed 123  -> ..._wf_<cfg>_seed123_<ts>.csv
  seed 2024 -> ..._wf_<cfg>_seed2024_<ts>.csv
Latest timestamp per (cfg, seed) is used. Missing (cfg,seed) is reported, not
guessed.

Run:  python scripts/analyze_multiseed.py
"""
from __future__ import annotations
import os, glob, re
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

CONFIGS = ["base", "irf", "all"]
SEEDS = ["42", "123", "2024"]


def find_csv(cfg: str, seed: str):
    if seed == "42":
        # no _seed suffix; exclude any _seedNNN to avoid matching other seeds
        cands = [p for p in glob.glob(
            f"data/processed/all_experiment_results_qwen2.5_wf_{cfg}_*.csv")
            if not re.search(r"_seed\d+_", p)]
    else:
        cands = glob.glob(
            f"data/processed/all_experiment_results_qwen2.5_wf_{cfg}_seed{seed}_*.csv")
    return max(cands, key=os.path.getmtime) if cands else None


def key(d):
    return (d["event_date"].astype(str).str[:10] + "|" +
            d["industry_code"].astype(str))


def main() -> int:
    print("=" * 72)
    print("Multi-seed aggregation  (seeds 42/123/2024)")
    print("=" * 72)
    store = {}   # cfg -> {seed -> df}
    for cfg in CONFIGS:
        store[cfg] = {}
        for s in SEEDS:
            p = find_csv(cfg, s)
            if p is None:
                print(f"  {cfg:<5} seed{s:<4}: MISSING")
                continue
            d = pd.read_csv(p); d["k"] = key(d)
            store[cfg][s] = d
            print(f"  {cfg:<5} seed{s:<4}: {os.path.basename(p)}  n={len(d)}")

    print("\n" + "=" * 72)
    print(f"{'config':<6}{'overall mean±std':>22}{'neg mean±std':>22}{'MAE mean':>12}{'maj-vote':>10}")
    print("=" * 72)
    agg = {}
    for cfg in CONFIGS:
        seeds_present = list(store[cfg].keys())
        if not seeds_present:
            print(f"{cfg:<6}  no data"); continue
        overalls, negs, maes = [], [], []
        for s in seeds_present:
            d = store[cfg][s]
            overalls.append(d["dir_correct"].mean())
            negs.append(d[d.real_dir == "-"]["dir_correct"].mean())
            maes.append(d["MAE"].mean())
        # majority vote over available seeds (per-event 2-of-3 correct)
        frames = [store[cfg][s][["k", "real_dir", "dir_correct"]] for s in seeds_present]
        m = frames[0].rename(columns={"dir_correct": "c0"})
        for i, f in enumerate(frames[1:], 1):
            m = m.merge(f[["k", "dir_correct"]].rename(columns={"dir_correct": f"c{i}"}), on="k")
        cols = [c for c in m.columns if c.startswith("c") and c != "correct"]
        votes = m[cols].astype(int).sum(axis=1)
        maj = (votes >= (len(cols) / 2.0)).mean()
        agg[cfg] = {"overall": np.array(overalls), "neg": np.array(negs),
                    "mae": np.array(maes), "maj": maj, "seeds": seeds_present}
        print(f"{cfg:<6}{np.mean(overalls)*100:>13.2f}±{np.std(overalls)*100:.2f}pp"
              f"{np.mean(negs)*100:>13.2f}±{np.std(negs)*100:.2f}pp"
              f"{np.mean(maes):>12.4f}{maj*100:>9.2f}%  (n_seed={len(seeds_present)})")

    if "base" in agg:
        print("\n" + "=" * 72)
        print("vs base (3-seed mean difference)")
        print("=" * 72)
        b = agg["base"]
        for cfg in ("irf", "all"):
            if cfg not in agg:
                continue
            a = agg[cfg]
            do = a["overall"].mean() - b["overall"].mean()
            dn = a["neg"].mean() - b["neg"].mean()
            dmae = a["mae"].mean() - b["mae"].mean()
            # CI separation heuristic: |Δ| vs pooled std
            pooled_o = np.sqrt(a["overall"].var() + b["overall"].var()) + 1e-9
            sep = abs(do) / (pooled_o + 1e-9)
            print(f"  [{cfg} vs base]  overall Δ={do*100:+.2f}pp  neg Δ={dn*100:+.2f}pp  "
                  f"MAE Δ={dmae:+.5f}  (|Δo|/pooled_sd={sep:.2f})")
    print("\n判读：方向看 mean±std 是否分离 + maj-vote；MAE 看 IRF −0.009 是否 3-seed 稳。")
    print("单seed噪声≈±2.4pp，3-seed 均值把噪声降到 ~±1.4pp（/√3）。")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
