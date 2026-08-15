r"""
v4.2 reviewer-response experiment batch  (ONE script, run it directly)
======================================================================
ROUTE 1 (health-check first): on the EXPANDED 96-event benchmark, run only the
3 core cells single-seed to confirm the expansion is healthy and the headline
leakage-bias survives, BEFORE spending GPU on debate / multi-seed.

    python scripts/run_v42_experiments.py            # 3 core cells + analysis (~20h)
    python scripts/run_v42_experiments.py --analyze-only

Core cells (expanded benchmark, seed 42):
    base    all firewall channels sealed (honest baseline)
    fx_111  == base (all sealed)  -- explicit factorial anchor
    fx_000  all channels leaky    -- the leakage-bias comparison anchor
The headline test is fx_111 (sealed) vs fx_000 (leaky): per-class optimism swing.

Safe to Ctrl-C and resume: each cell writes its own timestamped CSV; the analyzer
reads the latest per tag.

LEAKAGE / DATA NOTES
--------------------
* CAR_CSV points every child at the expanded 96-event benchmark
  (car_results_expanded.csv). Unset -> original 168 (we set it explicitly here).
* Every cell runs in a FRESH SUBPROCESS with firewall switches set EXPLICITLY
  (run_firewall_factorial._base_env) -- no env residue between cells.
* fx_000 is INTENTIONALLY leaky on all 3 channels -- that IS the experiment
  (the leakage-bias anchor). Every honest-system number stays firewalled.
* IRF is STOPPED (route-1 decision): the old leaky IRF and the leak-free/sentiment
  variants are NOT in this batch. Re-add later only if route-1 is healthy.
"""
from __future__ import annotations
import os, sys, subprocess, argparse, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = sys.executable
RUNNER = "scripts/run_firewall_factorial.py"
ANALYZER = "scripts/analyze_factorial.py"
SEED = "42"
CAR_CSV = "data/processed/car_results_expanded.csv"


def _run(argv: list[str]) -> int:
    """Run a child in a CLEAN env (no firewall/shrink/IRF residue), with CAR_CSV
    pinned to the expanded benchmark so every cell uses the 96-event set."""
    env = os.environ.copy()
    for k in ("FW_L1", "FW_L2", "FW_RAG", "SHRINK_MODE", "IRF_MODE", "IRF_LAMBDA",
              "IRF_LAMBDA_MAX", "USE_SENTIMENT", "SENT_BETA", "USE_IRF",
              "USE_FUNDAMENTALS", "USE_FUND_VALUATION", "ABLATION_MODE",
              "RUN_TAG", "LLM_MODEL", "USE_DEBATE"):
        env.pop(k, None)
    env["CAR_CSV"] = CAR_CSV
    print(f"\n>>> {' '.join(argv)}   [CAR_CSV={CAR_CSV}]", flush=True)
    return subprocess.run([PY] + argv, env=env).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(CAR_CSV):
        print(f"ERROR: {CAR_CSV} not found. Run scripts/expand_events.py + CAR first.")
        return 1

    t0 = datetime.datetime.now()
    print("=" * 72)
    print("v4.2 ROUTE-1 batch: expanded 96-event benchmark, 3 core cells, seed 42")
    print(f"start: {t0:%Y-%m-%d %H:%M:%S}   benchmark: {CAR_CSV}")
    print("=" * 72)

    if not args.analyze_only:
        # 3 core cells on the expanded benchmark (single seed). Tag suffix x96
        # keeps these CSVs from colliding with the old 168-benchmark runs.
        _run([RUNNER, "--which", "factorial", "--only", "fx_111", "fx_000",
              "--seed", SEED, "--tag-suffix", "x96"])
        _run([RUNNER, "--which", "irf", "--only", "base",
              "--seed", SEED, "--tag-suffix", "x96"])

    # analysis: fx_111x96 vs fx_000x96 per-class leakage swing on 96 events
    _run([ANALYZER, "--nboot", "5000", "--tag-suffix", "x96"])

    dt = (datetime.datetime.now() - t0).total_seconds() / 3600.0
    print("\n" + "=" * 72)
    print(f"route-1 batch done in {dt:.1f} h. Paste the analyzer output back to review.")
    print("If fx_111 vs fx_000 per-class swing holds on 96 events -> expand to debate + multi-seed.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
