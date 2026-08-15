r"""
Firewall 2^3 factorial + shrinkage controls (reviewers 5a & 2)
==============================================================

Two experiments, both as isolated subprocesses (env set explicitly, no residue).

(A) 2^3 FACTORIAL over the three firewall channels L1 (sensitivity prior),
    L2 (direction label), RAG (retrieval cutoff). Each channel 0=leaky/1=sealed.
    All-1 == the firewalled base (~60.5%); all-0 == full-data leaky (~61.9%).
    The 6 middle cells decompose the non-monotonicity the reviewer flagged
    (esp. clean-prior + dirty-RAG, i.e. L1=1,L2=1,RAG=0, ~ the 47.6% LOO regime).

(B) SHRINKAGE CONTROLS under the sealed firewall: does a LEAK-FREE generic
    shrink match IRF's MAE gain? Configs: base (no shrink), irf, and
    SHRINK_MODE in {neutral, fixed065, zero}. If neutral/fixed065 match irf's
    MAE, the DSGE transmission structure adds nothing beyond shrinkage.

Every subprocess is scripts/run_walk_forward.py with WALK_FORWARD=1 and the
per-run switches. Serial GPU (hours per cell). Seeds via --seed.

Usage (v4 dir):
    python scripts/run_firewall_factorial.py --which factorial   # (A) 8 cells
    python scripts/run_firewall_factorial.py --which shrink      # (B) 3 controls
    python scripts/run_firewall_factorial.py --which both --seed 42
    python scripts/run_firewall_factorial.py --summarize-only
"""
from __future__ import annotations
import os, sys, subprocess, argparse, glob, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# (A) 2^3 factorial: tag -> (FW_L1, FW_L2, FW_RAG)
FACTORIAL = {
    "fx_111": ("1", "1", "1"),   # all sealed  == firewalled base (~60.5)
    "fx_110": ("1", "1", "0"),   # clean prior+label, dirty RAG (~ LOO 47.6 regime)
    "fx_101": ("1", "0", "1"),   # clean prior+RAG, leaky label
    "fx_011": ("0", "1", "1"),   # leaky prior, clean label+RAG
    "fx_100": ("1", "0", "0"),
    "fx_010": ("0", "1", "0"),
    "fx_001": ("0", "0", "1"),
    "fx_000": ("0", "0", "0"),   # all leaky   == full-data (~61.9)
}

# (B) shrinkage controls: tag -> extra env (all under sealed firewall)
SHRINK = {
    "shrink_neutral":  {"SHRINK_MODE": "neutral"},   # net*=(1-λ), direction-free
    "shrink_fixed065": {"SHRINK_MODE": "fixed065"},  # net*=0.65
    "shrink_zero":     {"SHRINK_MODE": "zero"},      # pure TS (correction off)
}


def _base_env(seed: str) -> dict:
    env = os.environ.copy()
    env["WALK_FORWARD"] = "1"
    env["ABLATION_MODE"] = "none"
    env["USE_IRF"] = "0"
    env["USE_FUNDAMENTALS"] = "0"
    env["USE_FUND_VALUATION"] = "0"
    env["SEED"] = seed
    env.pop("LLM_MODEL", None)
    # clear any per-channel / shrink / IRF / sentiment / debate / reldecomp residue
    for k in ("FW_L1", "FW_L2", "FW_RAG", "SHRINK_MODE", "IRF_MODE",
              "IRF_LAMBDA", "IRF_LAMBDA_MAX", "USE_SENTIMENT", "SENT_BETA",
              "USE_DEBATE", "DEBATE_ABSTAIN", "DEBATE_ROUNDS", "USE_RELDECOMP"):
        env.pop(k, None)
    return env


def run_cell(tag: str, env: dict, model: str = "") -> int:
    env["RUN_TAG"] = tag
    if model:
        env["LLM_MODEL"] = model      # cross-model run; CSV filename carries the model tag
    banner = f"\n{'='*70}\n[factorial] {tag}  " + " ".join(
        f"{k}={env.get(k)}" for k in ("FW_L1", "FW_L2", "FW_RAG", "SHRINK_MODE", "USE_IRF", "SEED")
        if env.get(k) is not None) + f"\n{'='*70}"
    print(banner, flush=True)
    t0 = datetime.datetime.now()
    r = subprocess.run([sys.executable, "scripts/run_walk_forward.py"], env=env)
    dt = (datetime.datetime.now() - t0).total_seconds() / 60.0
    print(f"[factorial] {tag} done rc={r.returncode} in {dt:.1f} min", flush=True)
    return r.returncode


def latest_csv(tag: str) -> str | None:
    c = glob.glob(f"data/processed/all_experiment_results_qwen2.5_wf_{tag}_*.csv")
    return max(c, key=os.path.getmtime) if c else None


def summarize(tags: list[str]) -> None:
    import pandas as pd
    print(f"\n{'='*76}\n[factorial] summary (overall / pos / neg / MAE)\n{'='*76}")
    print(f"{'tag':<16}{'overall':>9}{'pos':>8}{'neg':>8}{'MAE':>9}  CSV")
    for tag in tags:
        p = latest_csv(tag)
        if not p:
            print(f"{tag:<16}{'MISSING':>9}"); continue
        d = pd.read_csv(p)
        acc = d["dir_correct"].mean()
        pos = d[d.real_dir == "+"]["dir_correct"].mean()
        neg = d[d.real_dir == "-"]["dir_correct"].mean()
        print(f"{tag:<16}{acc:>8.2%}{pos:>8.2%}{neg:>8.2%}{d['MAE'].mean():>9.4f}  "
              f"{os.path.basename(p)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["factorial", "shrink", "irf", "debate", "reldecomp", "both"], default="both")
    ap.add_argument("--seed", default="42")
    ap.add_argument("--only", nargs="*", help="subset of tags to run")
    ap.add_argument("--tag-suffix", default="",
                    help="append to each RUN_TAG (e.g. 'x96') so expanded-benchmark "
                         "CSVs do not collide with old-benchmark ones")
    ap.add_argument("--model", default="",
                    help="cross-model run: set LLM_MODEL for every cell (e.g. "
                         "llama3.1:8b). CSV filename carries the model tag.")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    sfx = args.tag_suffix

    plan: list[tuple[str, dict]] = []
    if args.which in ("factorial", "both"):
        for tag, (l1, l2, rag) in FACTORIAL.items():
            env = _base_env(args.seed)
            env["FW_L1"], env["FW_L2"], env["FW_RAG"] = l1, l2, rag
            plan.append((tag, env))
    if args.which in ("shrink", "both"):
        # reference cells for the shrink comparison
        env = _base_env(args.seed); plan.append(("base", env))
        env = _base_env(args.seed); env["USE_IRF"] = "1"; plan.append(("irf", env))
        for tag, extra in SHRINK.items():
            env = _base_env(args.seed); env.update(extra); plan.append((tag, env))
    if args.which == "irf":
        # IRF redesign (reviewer): base vs old leaky IRF (v1) vs leak-free v2
        # vs leak-free v2 + sentiment gate.
        env = _base_env(args.seed); plan.append(("base", env))
        env = _base_env(args.seed); env["USE_IRF"] = "1"; plan.append(("irf", env))
        env = _base_env(args.seed); env["USE_IRF"] = "1"
        env["IRF_MODE"] = "leakfree"; plan.append(("irf_lf", env))
        env = _base_env(args.seed); env["USE_IRF"] = "1"
        env["IRF_MODE"] = "leakfree"; env["USE_SENTIMENT"] = "1"
        plan.append(("irf_lf_sent", env))
    if args.which == "debate":
        # multi-agent debate (variant B): base (all firewall, no debate) vs debate.
        env = _base_env(args.seed); plan.append(("base", env))
        env = _base_env(args.seed); env["USE_DEBATE"] = "1"
        plan.append(("debate", env))
    if args.which == "reldecomp":
        # relative-return decomposition intervention: base vs reldecomp (3-step
        # R_ind / R_mkt / beats-market prompt). Tests the attribution causally.
        env = _base_env(args.seed); plan.append(("base", env))
        env = _base_env(args.seed); env["USE_RELDECOMP"] = "1"
        plan.append(("reldecomp", env))

    if args.only:
        plan = [(t, e) for (t, e) in plan if t in args.only]

    # append the suffix AFTER --only filtering, so callers select by base tag
    # but the CSVs/tags carry the suffix (e.g. fx_111 -> fx_111x96)
    plan = [(t + sfx, e) for (t, e) in plan]

    tags = [t for t, _ in plan]
    if not args.summarize_only:
        print(f"[factorial] running {len(plan)} cells serially: {tags}")
        for tag, env in plan:
            rc = run_cell(tag, env, model=args.model)
            if rc != 0:
                print(f"[factorial] {tag} FAILED rc={rc}, continuing", file=sys.stderr)
    summarize(tags)
    return 0


if __name__ == "__main__":
    sys.exit(main())
