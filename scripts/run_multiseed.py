"""
Multi-seed noise-reduction run (v4).

Runs base / irf / all for seeds 123 and 2024 (seed 42 already exists) to give
the direction metric a 3-seed CI and average out the ±2.4pp single-seed noise
that swamped every skill's effect. Verified real prediction is single-run only,
so we vary the SEED that drives random/np/torch seeding (main.py:58-61); the
per-seed CSV filenames carry a `_seed{N}` suffix (run_walk_forward.py:61-62) so
nothing overwrites the existing seed-42 products.

Isolation is DOUBLE: this script clears switch residue from its own env, and
run_ablation_matrix.py already runs each config in a fresh subprocess with all
five switches set explicitly. So no $env residue can leak in.

Time: 6 configs × ~3.5h ≈ 21h total (two seeds × three configs). Overnight ×2,
or one long run. Safe to re-run — run_walk_forward has RESUME_FROM (per config),
though this driver runs configs fresh.

Run (v4 dir, conda env):
    python scripts/run_multiseed.py
    python scripts/run_multiseed.py --seeds 123          # just one seed
    python scripts/run_multiseed.py --configs base irf   # subset of configs
"""
from __future__ import annotations
import os, sys, subprocess, argparse, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

SWITCHES = ["WALK_FORWARD", "USE_IRF", "USE_FUNDAMENTALS",
            "USE_FUND_VALUATION", "ABLATION_MODE", "SEED", "RUN_TAG"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", default=["123", "2024"],
                    help="seeds to run (42 already exists; default 123 2024)")
    ap.add_argument("--configs", nargs="*", default=["base", "irf", "all"],
                    help="ablation configs per seed (default base irf all)")
    args = ap.parse_args()

    # clean env: strip any switch residue so the child ablation-matrix starts
    # from a known state (it re-sets everything explicitly per config anyway).
    base_env = os.environ.copy()
    for k in SWITCHES:
        base_env.pop(k, None)

    t0 = datetime.datetime.now()
    print(f"[multiseed] seeds={args.seeds} configs={args.configs} "
          f"start {t0:%Y-%m-%d %H:%M}")
    print(f"[multiseed] 预计 {len(args.seeds)*len(args.configs)} 次全量 × ~3.5h "
          f"≈ {len(args.seeds)*len(args.configs)*3.5:.0f}h\n")

    for seed in args.seeds:
        banner = f"\n{'#'*72}\n[multiseed] ===== SEED {seed} =====\n{'#'*72}"
        print(banner, flush=True)
        cmd = [sys.executable, "scripts/run_ablation_matrix.py",
               "--only", *args.configs, "--seed", str(seed)]
        print(f"[multiseed] $ {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, env=base_env)
        print(f"[multiseed] SEED {seed} 完成 rc={r.returncode}", flush=True)

    dt = (datetime.datetime.now() - t0).total_seconds() / 3600.0
    print(f"\n{'='*72}")
    print(f"[multiseed] 全部完成，用时 {dt:.1f}h")
    print(f"[multiseed] 现在三个 seed 齐（42 已有 + {' '.join(args.seeds)}）。")
    print(f"[multiseed] 下一步：python scripts/analyze_multiseed.py  出 3-seed CI + MAE 均值")
    print(f"{'='*72}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
