r"""
v4.3 reviewer-response GPU batch  (ONE script, run it directly).
================================================================
    python scripts/run_v43_experiments.py

Addresses the reviewer's acceptance prerequisites that need GPU:
  P1  reldecomp seed 123 + 2024   -> 3-seed the mechanism result (#3)
  P2  the 6 MIDDLE 2^3 factorial cells (single seed) -> full factorial, not just
      the two endpoints, so no reviewer can claim one channel drives everything (#4)
  P3  a SECOND model (llama3.1:8b) on the key cells -> show near-random + optimism
      bias is not a Qwen artifact; supports the generalization claim (#7)

Serial GPU. Rough time: P1 ~7h (2 x 321), P2 ~21h (6 x 321), P3 ~10h
(base + fx_000 + reldecomp on an 8B model, faster per call). Total ~1.5-2 days.
Safe to Ctrl-C: each cell writes its own timestamped CSV; re-running only redoes
missing cells if you comment out finished phases.

LEAKAGE SAFETY (verified pattern, identical to run_v42_multiseed)
-----------------------------------------------------------------
* Each child runs in a FRESH env: every firewall/shrink/IRF/sentiment/debate/
  reldecomp switch is STRIPPED from the inherited environment before launch;
  the child (run_firewall_factorial) sets exactly what each cell needs.
* CAR_CSV pinned to the expanded 96-event benchmark for every child.
* P2's 6 middle cells are INTENTIONALLY channel-leaky -- that IS the factorial
  (measuring each channel's main effect + interactions). Endpoints fx_111/fx_000
  already exist (seed 42) so are not re-run here.
* P3 sets --model llama3.1:8b; the CSV filename carries the model tag
  (llama3.1_...) so cross-model runs never collide with qwen runs.
"""
from __future__ import annotations
import os, sys, subprocess, datetime, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = sys.executable
RUNNER = "scripts/run_firewall_factorial.py"
ANALYZER = "scripts/analyze_factorial.py"
CAR_CSV = "data/processed/car_results_expanded.csv"
MODEL2 = "llama3.1:8b"           # second model (edit here to swap, e.g. deepseek-r1:32b)

_SCRUB = ("FW_L1", "FW_L2", "FW_RAG", "SHRINK_MODE", "IRF_MODE", "IRF_LAMBDA",
          "IRF_LAMBDA_MAX", "USE_SENTIMENT", "SENT_BETA", "USE_IRF",
          "USE_FUNDAMENTALS", "USE_FUND_VALUATION", "ABLATION_MODE",
          "RUN_TAG", "USE_DEBATE", "DEBATE_ABSTAIN", "DEBATE_ROUNDS",
          "USE_RELDECOMP", "WALK_FORWARD", "SEED", "LLM_MODEL")


def _run(argv):
    """Run a child in a scrubbed env with CAR_CSV pinned. The child sets its own
    switches; nothing leaks from the parent shell."""
    env = os.environ.copy()
    for k in _SCRUB:
        env.pop(k, None)
    env["CAR_CSV"] = CAR_CSV
    print(f"\n>>> {' '.join(argv)}   [CAR_CSV={CAR_CSV}]", flush=True)
    return subprocess.run([PY] + argv, env=env).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-p1", action="store_true", help="skip reldecomp multi-seed")
    ap.add_argument("--skip-p2", action="store_true", help="skip 6 middle factorial cells")
    ap.add_argument("--skip-p3", action="store_true", help="skip second-model run")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(CAR_CSV):
        print(f"ERROR: {CAR_CSV} missing."); return 1

    t0 = datetime.datetime.now()
    print("=" * 72)
    print("v4.3 GPU batch  (expanded 96-event benchmark)")
    print(f"start: {t0:%Y-%m-%d %H:%M:%S}")
    print("=" * 72)

    if not args.analyze_only:
        # P1: reldecomp seed 123 + 2024 (seed 42 already done) -> 3-seed mechanism
        if not args.skip_p1:
            for seed in ("123", "2024"):
                _run([RUNNER, "--which", "reldecomp", "--only", "reldecomp",
                      "--seed", seed, "--tag-suffix", "x96"])
        # P2: the 6 MIDDLE factorial cells (single seed 42). Endpoints fx_111/
        # fx_000 already exist. --which factorial builds all 8; --only selects the 6.
        if not args.skip_p2:
            _run([RUNNER, "--which", "factorial", "--only",
                  "fx_110", "fx_101", "fx_011", "fx_100", "fx_010", "fx_001",
                  "--seed", "42", "--tag-suffix", "x96"])
        # P3: second model (llama3.1:8b) on key cells -> generalization.
        # Uses the 'irf' plan's base cell + factorial endpoints, model-tagged CSVs.
        if not args.skip_p3:
            _run([RUNNER, "--which", "factorial", "--only", "fx_111", "fx_000",
                  "--seed", "42", "--tag-suffix", "x96m2", "--model", MODEL2])
            _run([RUNNER, "--which", "reldecomp", "--only", "reldecomp",
                  "--seed", "42", "--tag-suffix", "x96m2", "--model", MODEL2])

    # analysis on the primary (qwen) expanded runs
    _run([ANALYZER, "--nboot", "5000", "--tag-suffix", "x96"])

    dt = (datetime.datetime.now() - t0).total_seconds() / 3600.0
    print("\n" + "=" * 72)
    print(f"v4.3 batch done in {dt:.1f} h.")
    print("Paste back: (a) the analyzer output, (b) paths of new reldecompx96 /")
    print("fx_*x96 (6 middle) / *x96m2 (llama) CSVs. I'll aggregate + write the paper.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
