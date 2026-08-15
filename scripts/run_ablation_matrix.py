"""
v4 ablation matrix runner (walk-forward, leak-free)
===================================================

Runs the component-ablation matrix as INDEPENDENT SUBPROCESSES so env-var
residue can never leak between configs (the $env: contamination bug that once
made an A3-tagged run pose as cross-LLM). Every subprocess gets a freshly
constructed environment where ALL four switches are set EXPLICITLY — nothing
is inherited, nothing is assumed.

Configs (all walk-forward = firewalled; the honest regime):
    base   WALK_FORWARD=1  IRF=0 FUND=0    -> Phase-0 honest baseline
    irf    WALK_FORWARD=1  IRF=1 FUND=0    -> + transmission IRF prior
    fund   WALK_FORWARD=1  IRF=0 FUND=1    -> + price-derived fundamentals
    all    WALK_FORWARD=1  IRF=1 FUND=1    -> both (synergy)

Each writes data/processed/all_experiment_results_qwen2.5_wf_<tag>_<ts>.csv.
GPU work is serial (one config at a time). ~hours per config.

Usage (PowerShell / bash, in v4 dir):
    python scripts/run_ablation_matrix.py
    python scripts/run_ablation_matrix.py --only irf all      # subset
    python scripts/run_ablation_matrix.py --seed 42
After it finishes it prints a comparison table and runs the VaR backtest per CSV.
"""
from __future__ import annotations
import os, sys, subprocess, argparse, glob, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

CONFIGS = {
    "base":    {"WALK_FORWARD": "1", "USE_IRF": "0", "USE_FUNDAMENTALS": "0", "USE_FUND_VALUATION": "0"},
    "irf":     {"WALK_FORWARD": "1", "USE_IRF": "1", "USE_FUNDAMENTALS": "0", "USE_FUND_VALUATION": "0"},
    "fund":    {"WALK_FORWARD": "1", "USE_IRF": "0", "USE_FUNDAMENTALS": "1", "USE_FUND_VALUATION": "0"},
    "fundval": {"WALK_FORWARD": "1", "USE_IRF": "0", "USE_FUNDAMENTALS": "0", "USE_FUND_VALUATION": "1"},
    "all":     {"WALK_FORWARD": "1", "USE_IRF": "1", "USE_FUNDAMENTALS": "1", "USE_FUND_VALUATION": "1"},
}


def run_config(tag: str, switches: dict, seed: str) -> int:
    # fresh env: inherit PATH/CUDA/etc, then set EVERY switch explicitly so no
    # residue from a previous config or the parent shell can leak in.
    env = os.environ.copy()
    env["ABLATION_MODE"] = "none"       # explicit: not an A1/A2/A3 ablation
    env["WALK_FORWARD"] = switches["WALK_FORWARD"]
    env["USE_IRF"] = switches["USE_IRF"]
    env["USE_FUNDAMENTALS"] = switches["USE_FUNDAMENTALS"]
    env["USE_FUND_VALUATION"] = switches.get("USE_FUND_VALUATION", "0")  # 显式,防残留
    env["SEED"] = seed
    env["RUN_TAG"] = tag
    env.pop("LLM_MODEL", None)          # default qwen2.5:32b; not a cross-LLM run

    banner = (f"\n{'='*70}\n[ablation-matrix] CONFIG={tag}  "
              f"WALK_FORWARD={env['WALK_FORWARD']} USE_IRF={env['USE_IRF']} "
              f"USE_FUNDAMENTALS={env['USE_FUNDAMENTALS']} SEED={seed}\n{'='*70}")
    print(banner, flush=True)
    t0 = datetime.datetime.now()
    r = subprocess.run([sys.executable, "scripts/run_walk_forward.py"], env=env)
    dt = (datetime.datetime.now() - t0).total_seconds() / 60.0
    print(f"[ablation-matrix] CONFIG={tag} done rc={r.returncode} in {dt:.1f} min",
          flush=True)
    return r.returncode


def latest_csv(tag: str) -> str | None:
    pat = f"data/processed/all_experiment_results_qwen2.5_wf_{tag}_*.csv"
    c = glob.glob(pat)
    return max(c, key=os.path.getmtime) if c else None


def summarize(tags: list[str]) -> None:
    import pandas as pd
    print(f"\n{'='*70}\n[ablation-matrix] 消融对比（walk-forward, 全 firewall）\n{'='*70}")
    print(f"{'config':<8}{'overall':>10}{'pos':>10}{'neg':>10}{'MAE':>10}  CSV")
    base_neg = None
    for tag in tags:
        p = latest_csv(tag)
        if not p:
            print(f"{tag:<8}{'MISSING':>10}")
            continue
        d = pd.read_csv(p)
        acc = d["dir_correct"].mean()
        pos = d[d.real_dir == "+"]["dir_correct"].mean()
        neg = d[d.real_dir == "-"]["dir_correct"].mean()
        mae = d["MAE"].mean()
        if tag == "base":
            base_neg = neg
        flag = ""
        if base_neg is not None and tag != "base":
            flag = f"  (neg {'+' if neg>=base_neg else ''}{(neg-base_neg)*100:.1f}pp vs base)"
        print(f"{tag:<8}{acc:>9.2%}{pos:>9.2%}{neg:>9.2%}{mae:>10.4f}  "
              f"{os.path.basename(p)}{flag}")
    print("\n关键看 neg 列：Phase-0 base 负向≈50%，IRF/基本面若有效应把它抬高。")
    print("每个 CSV 可跑：python scripts/analyze_risk.py <csv> 出 VaR 回测。")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(CONFIGS),
                    help="只跑这些配置（默认全部）")
    ap.add_argument("--seed", default="42")
    ap.add_argument("--summarize-only", action="store_true",
                    help="不跑实验，只汇总已有 CSV")
    args = ap.parse_args()

    tags = args.only if args.only else list(CONFIGS)
    if not args.summarize_only:
        for tag in tags:
            rc = run_config(tag, CONFIGS[tag], args.seed)
            if rc != 0:
                print(f"[ablation-matrix] CONFIG={tag} 失败 rc={rc}，继续下一个",
                      file=sys.stderr)
    summarize(tags)
    return 0


if __name__ == "__main__":
    sys.exit(main())
