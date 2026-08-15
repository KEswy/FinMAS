r"""
v4.2 multi-seed + LLM-only batch on the EXPANDED 96-event benchmark.
====================================================================
Run it directly:
    python scripts/run_v42_multiseed.py
    python scripts/run_v42_multiseed.py --analyze-only

Locks down the X-positioning headline (leakage-induced optimism bias) with THREE
seeds, and gets an LLM-only baseline on the 96-event set.

Cells (serial GPU; fx cell ~7h on 321 scenarios, LLM-only ~4h):
  1. fx_111 + fx_000, seed 123   (--tag-suffix x96)
  2. fx_111 + fx_000, seed 2024  (--tag-suffix x96)
  3. LLM-only direct prompting on car_results_expanded.csv
  4. analysis at --tag-suffix x96 (headline leakage-bias, now 3-seed capable)
seed 42 for fx_111x96/fx_000x96 already exists from the route-1 run.

LEAKAGE SAFETY (verified, read once)
------------------------------------
* Every child runs in a FRESH env: we STRIP all firewall/shrink/IRF/sentiment/
  ablation switches from the inherited environment, then the child sets exactly
  what it needs. No residue can cross between cells (the old A2/A3 $env bug).
* CAR_CSV is pinned to the expanded benchmark for every child, so all cells use
  the same 96-event ground truth.
* fx_000 is INTENTIONALLY leaky on all 3 channels -- that IS the experiment
  (the leakage anchor). Every honest-system number (fx_111) stays firewalled.
* LLM-only reads only event text (no prior/label/retrieval) -> firewall-clean by
  construction; it takes no firewall switches.
"""
from __future__ import annotations
import os, sys, subprocess, argparse, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = sys.executable
RUNNER = "scripts/run_firewall_factorial.py"
ANALYZER = "scripts/analyze_factorial.py"
LLMONLY = "scripts/baseline_llm_only.py"
CAR_CSV = "data/processed/car_results_expanded.csv"

# every switch that could bias a cell if inherited from the parent shell
_SCRUB = ("FW_L1", "FW_L2", "FW_RAG", "SHRINK_MODE", "IRF_MODE", "IRF_LAMBDA",
          "IRF_LAMBDA_MAX", "USE_SENTIMENT", "SENT_BETA", "USE_IRF",
          "USE_FUNDAMENTALS", "USE_FUND_VALUATION", "ABLATION_MODE",
          "RUN_TAG", "USE_DEBATE", "DEBATE_ABSTAIN", "DEBATE_ROUNDS",
          "WALK_FORWARD", "SEED")


def _run(argv, extra_env=None):
    """Run a child in a scrubbed env with CAR_CSV pinned. extra_env adds only
    what THIS call needs (e.g. LLM_MODEL); it never leaks to the next call."""
    env = os.environ.copy()
    for k in _SCRUB:
        env.pop(k, None)
    env["CAR_CSV"] = CAR_CSV
    env.pop("LLM_MODEL", None)          # default qwen2.5:32b unless extra_env sets it
    if extra_env:
        env.update(extra_env)
    print(f"\n>>> {' '.join(argv)}   [CAR_CSV={CAR_CSV}"
          + (f", {extra_env}]" if extra_env else "]"), flush=True)
    return subprocess.run([PY] + argv, env=env).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--include-seeds", action="store_true",
                    help="also (re)run seed123/2024 factorial + LLM-only; omit if "
                         "those already ran (saves ~14h)")
    args = ap.parse_args()

    if not os.path.exists(CAR_CSV):
        print(f"ERROR: {CAR_CSV} missing; run expand_events + CAR first."); return 1

    t0 = datetime.datetime.now()
    print("=" * 72)
    print("v4.2 multi-seed + LLM-only  (expanded 96-event benchmark)")
    print(f"start: {t0:%Y-%m-%d %H:%M:%S}")
    print("=" * 72)

    if not args.analyze_only:
        if args.include_seeds:
            # multi-seed factorial anchors (seed 42 done) -- SKIP if already run.
            for seed in ("123", "2024"):
                _run([RUNNER, "--which", "factorial", "--only", "fx_111", "fx_000",
                      "--seed", seed, "--tag-suffix", "x96"])
            _run([LLMONLY, "--csv-target", CAR_CSV])
        # NEW interventions (this is what remains to run):
        # A. relative-return decomposition -- the key causal test of the
        #    attribution (3-step R_ind / R_mkt / beats-market prompt).
        _run([RUNNER, "--which", "reldecomp", "--only", "reldecomp",
              "--seed", "42", "--tag-suffix", "x96"])
        # B. debate variant B with CONTESTED-based abstention (the old
        #    confidence gate fired 319/321; tag x96b to keep the old run).
        _run([RUNNER, "--which", "debate", "--only", "debate",
              "--seed", "42", "--tag-suffix", "x96b"])

    _run([ANALYZER, "--nboot", "5000", "--tag-suffix", "x96"])

    dt = (datetime.datetime.now() - t0).total_seconds() / 3600.0
    print("\n" + "=" * 72)
    print(f"multi-seed batch done in {dt:.1f} h. Paste the (0) HEADLINE block back.")
    print("Now fx_111x96/fx_000x96 have seeds 42/123/2024 -> headline is 3-seed.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
