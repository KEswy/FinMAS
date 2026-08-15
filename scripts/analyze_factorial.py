r"""
Analyze the firewall factorial + shrinkage controls + run-to-run noise.
=======================================================================
Consumes the CSVs written by run_firewall_factorial.py. Three reports:

(1) NOISE band: all fx_111_*.csv (+ base_*.csv) same config, different runs ->
    run-to-run overall/neg mean/std/min/max. Honest replacement for the
    "0.28pp stable" claim.
(2) FACTORIAL: the 8 fx_XXX cells (overall/pos/neg/MAE) + per-channel marginal
    effect of SEALING each of L1/L2/RAG (mean Delta over the paired cells).
(3) SHRINK controls: base / irf / shrink_{neutral,fixed065,zero} MAE, with an
    EVENT-CLUSTER bootstrap CI on the MAE delta vs base and vs irf -- answers
    reviewer #2 (does leak-free generic shrink match the IRF magnitude gain?).

CLI:  python scripts/analyze_factorial.py [--nboot 5000]
"""
from __future__ import annotations
import os, sys, io, glob, argparse
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

PROC = "data/processed"
SUFFIX = ""   # set by --tag-suffix; expanded-benchmark runs use "x96"


MIN_ROWS = 300   # a full run is 321 rows; skip crashed/partial CSVs (e.g. n=12)


def _load(tag):
    """Latest COMPLETE CSV for a run tag (+ global SUFFIX), or None. Partial
    files (an interrupted run that was re-run) are skipped so they cannot
    pollute the aggregate."""
    tag = tag + SUFFIX
    c = glob.glob(f"{PROC}/all_experiment_results_qwen2.5_wf_{tag}_*.csv")
    # newest first; take the newest file that is complete
    for f in sorted(c, key=os.path.getmtime, reverse=True):
        d = pd.read_csv(f)
        if len(d) >= MIN_ROWS:
            d["event_date"] = pd.to_datetime(d["event_date"]).dt.strftime("%Y-%m-%d")
            return d
    return None


def _cls(d):
    ov = d["dir_correct"].mean()
    pos = d[d.real_dir == "+"]["dir_correct"].mean()
    neg = d[d.real_dir == "-"]["dir_correct"].mean()
    return ov, pos, neg, d["MAE"].mean()


def noise_band():
    print("\n" + "=" * 70 + "\n(1) RUN-TO-RUN NOISE (same config, different runs)\n" + "=" * 70)
    accs, negs = [], []
    for pat in ("fx_111" + SUFFIX, "base" + SUFFIX):
        for f in glob.glob(f"{PROC}/all_experiment_results_qwen2.5_wf_{pat}_*.csv"):
            d = pd.read_csv(f)
            if len(d) < MIN_ROWS:      # skip crashed/partial runs (e.g. n=12)
                continue
            accs.append(d["dir_correct"].mean())
            negs.append(d[d.real_dir == "-"]["dir_correct"].mean())
    if not accs:
        print("  (no fx_111/base CSVs yet)"); return
    a = np.array(accs); n = np.array(negs)
    print(f"  runs={len(a)}  overall mean={a.mean()*100:.2f}% std={a.std(ddof=1)*100:.2f}pp "
          f"range=[{a.min()*100:.2f},{a.max()*100:.2f}]")
    print(f"            neg mean={n.mean()*100:.2f}% std={n.std(ddof=1)*100:.2f}pp "
          f"range=[{n.min()*100:.2f},{n.max()*100:.2f}]")
    print("  -> report this band, not the 3-seed 0.28; headline uses 3-seed majority vote.")


FACT = ["fx_111", "fx_110", "fx_101", "fx_011", "fx_100", "fx_010", "fx_001", "fx_000"]


def factorial():
    print("\n" + "=" * 70 + "\n(2) 2^3 FIREWALL FACTORIAL (1=sealed, 0=leaky; order L1L2RAG)\n" + "=" * 70)
    got = {}
    print(f"  {'cell':<8}{'L1 L2 RAG':<11}{'overall':>9}{'pos':>8}{'neg':>8}{'MAE':>9}")
    for tag in FACT:
        d = _load(tag)
        if d is None:
            print(f"  {tag:<8}{'':<11}{'MISSING':>9}"); continue
        bits = tag.split("_")[1]
        got[bits] = _cls(d)
        ov, pos, neg, mae = got[bits]
        lab = " ".join(bits)
        print(f"  {tag:<8}{lab:<11}{ov*100:>8.2f}{pos*100:>8.2f}{neg*100:>8.2f}{mae:>9.4f}")
    # marginal effect of SEALING each channel: mean(overall|bit=1) - mean(overall|bit=0)
    if len(got) >= 4:
        print("\n  Marginal effect of SEALING a channel (mean Δoverall, pp):")
        for pos_i, name in enumerate(["L1", "L2", "RAG"]):
            sealed = [v[0] for k, v in got.items() if k[pos_i] == "1"]
            leaky = [v[0] for k, v in got.items() if k[pos_i] == "0"]
            if sealed and leaky:
                print(f"    {name}: {(np.mean(sealed)-np.mean(leaky))*100:+.2f}pp "
                      f"(sealed n={len(sealed)}, leaky n={len(leaky)})")


def _cluster_boot_mae(a, b, ev, nboot, rng):
    """Bootstrap CI for mean(MAE_a) - mean(MAE_b), resampling by event cluster."""
    clusters = pd.unique(ev)
    idx_by = {c: np.where(ev == c)[0] for c in clusters}
    deltas = []
    for _ in range(nboot):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        ii = np.concatenate([idx_by[c] for c in pick])
        deltas.append(a[ii].mean() - b[ii].mean())
    d = np.array(deltas)
    return float(np.mean(a) - np.mean(b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def shrink(nboot):
    print("\n" + "=" * 70 + "\n(3) SHRINKAGE CONTROLS (MAE; event-cluster bootstrap CI)\n" + "=" * 70)
    tags = ["base", "irf", "shrink_neutral", "shrink_fixed065", "shrink_zero"]
    data = {t: _load(t) for t in tags}
    base = data.get("base")
    if base is None:
        print("  (base missing)"); return
    rng = np.random.default_rng(42)
    ev = base["event_date"].values
    bmae = base["MAE"].values
    irf = data.get("irf")
    imae = irf["MAE"].values if irf is not None else None
    print(f"  {'config':<16}{'MAE':>9}{'ΔvsBase [95% CI]':>26}{'ΔvsIRF [95% CI]':>26}")
    for t in tags:
        d = data[t]
        if d is None:
            print(f"  {t:<16}{'MISSING':>9}"); continue
        m = d["MAE"].values
        # align on shared (event_date,industry) if lengths differ; assume same order/len
        row = f"  {t:<16}{m.mean():>9.4f}"
        if t != "base":
            md, lo, hi = _cluster_boot_mae(m, bmae, ev, nboot, rng)
            row += f"   {md:+.4f} [{lo:+.4f},{hi:+.4f}]"
        else:
            row += f"{'—':>26}"
        if imae is not None and t not in ("base", "irf"):
            md2, lo2, hi2 = _cluster_boot_mae(m, imae, ev, nboot, rng)
            row += f"  {md2:+.4f} [{lo2:+.4f},{hi2:+.4f}]"
        print(row)
    print("\n  READ: if shrink_neutral/fixed065 ΔvsIRF CI includes 0 -> generic")
    print("        shrink MATCHES IRF => transmission structure adds nothing (drop IRF).")
    print("        if IRF clearly lower (shrink ΔvsIRF CI all >0) -> keep IRF + disclose 5d leak.")


def _cluster_boot_acc(correct, ev, correct_b, nboot, rng):
    """Event-cluster bootstrap CI for mean(acc_a) - mean(acc_b) (paired by row)."""
    clusters = pd.unique(ev)
    idx_by = {c: np.where(ev == c)[0] for c in clusters}
    d = []
    for _ in range(nboot):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        ii = np.concatenate([idx_by[c] for c in pick])
        d.append(correct[ii].mean() - correct_b[ii].mean())
    d = np.array(d)
    return float(correct.mean() - correct_b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def irf_redesign(nboot):
    """DIRECTION comparison base vs v1-IRF vs leak-free v2-IRF (the redesign).
    Direction (esp. negative class) is the metric IRF can win on; MAE cannot
    (zero-correction already wins MAE)."""
    print("\n" + "=" * 70 + "\n(4) IRF REDESIGN + SENTIMENT: DIRECTION\n"
          "    base vs v1 vs leak-free v2 vs v2+sentiment-gate\n" + "=" * 70)
    tags = ["base", "irf", "irf_lf", "irf_lf_sent"]
    data = {t: _load(t) for t in tags}
    b = data.get("base")
    if b is None:
        print("  (base missing)"); return
    # align all on base's (event_date,industry) order
    key = ["event_date", "industry_code"]
    b = b.copy(); b["industry_code"] = b["industry_code"].astype(str)
    ev = b["event_date"].values
    rng = np.random.default_rng(7)
    print(f"  {'config':<10}{'overall':>9}{'pos':>8}{'neg':>8}"
          f"{'Δneg vs base [95% CI]':>28}")
    for t in tags:
        d = data[t]
        if d is None:
            print(f"  {t:<10}{'MISSING':>9}"); continue
        d = d.copy(); d["industry_code"] = d["industry_code"].astype(str)
        m = b[key].merge(d[key + ["dir_correct", "real_dir"]], on=key, how="left")
        ov, pos, neg, _ = _cls(d)
        row = f"  {t:<10}{ov*100:>8.2f}{pos*100:>8.2f}{neg*100:>8.2f}"
        if t != "base":
            # negative-class accuracy delta vs base, event-cluster bootstrap
            negmask = (b["real_dir"] == "-").values
            ca = m["dir_correct"].fillna(0).values.astype(float)[negmask]
            cb = b["dir_correct"].values.astype(float)[negmask]
            evn = ev[negmask]
            md, lo, hi = _cluster_boot_acc(ca, evn, cb, nboot, rng)
            row += f"   {md*100:+.2f}pp [{lo*100:+.2f},{hi*100:+.2f}]"
        print(row)
    print("\n  READ: leak-free v2 wins IFF it lifts NEG accuracy with CI excluding 0")
    print("        (generic shrinkage cannot move direction at all). If CI includes")
    print("        0 -> honest negative result, but the 5d leak is gone either way.")


def leakage_bias(nboot):
    """HEADLINE: fx_111 (sealed) vs fx_000 (leaky), per-class, event-cluster
    bootstrap. Tests whether leakage installs an optimism bias rather than
    inflating overall accuracy."""
    print("\n" + "=" * 70 + "\n(0) HEADLINE: leakage-bias  fx_111(sealed) vs fx_000(leaky)\n" + "=" * 70)
    a = _load("fx_111"); b = _load("fx_000")
    if a is None or b is None:
        print("  (need both fx_111 and fx_000 for this benchmark)"); return
    key = ["event_date", "industry_code"]
    for d in (a, b):
        d["industry_code"] = d["industry_code"].astype(str)
    m = a[key + ["real_dir", "dir_correct"]].merge(
        b[key + ["dir_correct"]], on=key, how="inner", suffixes=("_seal", "_leak"))
    ev = m["event_date"].values
    rd = m["real_dir"].values
    csl = m["dir_correct_seal"].astype(float).values
    clk = m["dir_correct_leak"].astype(float).values
    rng = np.random.default_rng(0)
    print(f"  n={len(m)} scenarios, {len(pd.unique(ev))} event clusters")
    print(f"  {'class':<9}{'sealed':>8}{'leaky':>8}{'Δ(seal-leak)':>14}{'  95% CI':>20}{'  p':>8}")
    for name, mask in [("overall", np.ones(len(rd), bool)),
                       ("positive", rd == "+"), ("negative", rd == "-")]:
        cl = pd.unique(ev[mask]); by = {c: np.where((ev == c) & mask)[0] for c in cl}
        g = csl[mask].mean() - clk[mask].mean()
        ds = []
        for _ in range(nboot):
            pk = rng.choice(cl, len(cl), True)
            ii = np.concatenate([by[c] for c in pk])
            ds.append(csl[ii].mean() - clk[ii].mean())
        ds = np.array(ds); p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
        lo, hi = np.percentile(ds, 2.5), np.percentile(ds, 97.5)
        print(f"  {name:<9}{csl[mask].mean()*100:>7.1f}%{clk[mask].mean()*100:>7.1f}%"
              f"{g*100:>+13.2f}pp  [{lo*100:+.1f},{hi*100:+.1f}]{p:>8.4f}")
    print("  READ: overall Δ~0 but negative-class Δ>0 with p<0.05 => leakage installs")
    print("        an optimism bias (kills neg-class), firewall recovers it. HEADLINE.")


def main():
    global SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=5000)
    ap.add_argument("--tag-suffix", default="",
                    help="read CSVs with this tag suffix (e.g. 'x96' for expanded benchmark)")
    a = ap.parse_args()
    SUFFIX = a.tag_suffix
    print(f"[analyze] tag suffix = '{SUFFIX or '(none)'}'")
    leakage_bias(a.nboot)
    noise_band()
    factorial()
    shrink(a.nboot)
    irf_redesign(a.nboot)


if __name__ == "__main__":
    main()
