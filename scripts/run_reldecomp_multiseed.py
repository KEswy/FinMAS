r"""
Reldecomp intervention -- multi-seed top-up (seed 42 already done).
==================================================================
Run it directly:
    python scripts/run_reldecomp_multiseed.py

Adds seed 123 and 2024 for the relative-return decomposition intervention so the
mechanism result (absolute-vs-relative misalignment 68.4% -> 39.2%) is 3-seed,
matching the headline. ~7h GPU (two 321-scenario runs, serial).

LEAKAGE SAFETY (verified)
-------------------------
* Each child runs in a FRESH env: every firewall/shrink/IRF/sentiment/debate/
  reldecomp switch is STRIPPED from the inherited environment, then the child
  (run_firewall_factorial, --which reldecomp --only reldecomp) sets exactly
  USE_RELDECOMP=1 itself. A dirty parent shell cannot bias a cell.
* CAR_CSV is pinned to the expanded 96-event benchmark for every child.
* --only reldecomp runs ONLY the reldecomp cell (not the base reference, which
  already exists per seed from the factorial runs); --tag-suffix x96 keeps the
  filenames aligned with the rest of the expanded-benchmark runs.
"""
from __future__ import annotations
import os, sys, subprocess, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = sys.executable
RUNNER = "scripts/run_firewall_factorial.py"
ANALYZER = "scripts/analyze_factorial.py"
CAR_CSV = "data/processed/car_results_expanded.csv"

_SCRUB = ("FW_L1", "FW_L2", "FW_RAG", "SHRINK_MODE", "IRF_MODE", "IRF_LAMBDA",
          "IRF_LAMBDA_MAX", "USE_SENTIMENT", "SENT_BETA", "USE_IRF",
          "USE_FUNDAMENTALS", "USE_FUND_VALUATION", "ABLATION_MODE",
          "RUN_TAG", "USE_DEBATE", "DEBATE_ABSTAIN", "DEBATE_ROUNDS",
          "USE_RELDECOMP", "WALK_FORWARD", "SEED")


def _run(argv):
    env = os.environ.copy()
    for k in _SCRUB:
        env.pop(k, None)
    env["CAR_CSV"] = CAR_CSV
    env.pop("LLM_MODEL", None)
    print(f"\n>>> {' '.join(argv)}   [CAR_CSV={CAR_CSV}]", flush=True)
    return subprocess.run([PY] + argv, env=env).returncode


def main() -> int:
    if not os.path.exists(CAR_CSV):
        print(f"ERROR: {CAR_CSV} missing."); return 1
    t0 = datetime.datetime.now()
    print("=" * 72)
    print("reldecomp multi-seed top-up (seed 123 + 2024)")
    print(f"start: {t0:%Y-%m-%d %H:%M:%S}")
    print("=" * 72)
    for seed in ("123", "2024"):
        _run([RUNNER, "--which", "reldecomp", "--only", "reldecomp",
              "--seed", seed, "--tag-suffix", "x96"])
    dt = (datetime.datetime.now() - t0).total_seconds() / 3600.0
    print("\n" + "=" * 72)
    print(f"done in {dt:.1f} h. reldecompx96 now has seeds 42/123/2024.")
    print("Paste the 3 reldecompx96 CSV paths back; I'll aggregate the mechanism")
    print("metric (misalignment follow-abs 68.4% -> 39.2%) across seeds.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
