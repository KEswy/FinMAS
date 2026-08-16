"""Run a deterministic walk-forward evaluation for FinMAS v5 decision features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from finmas.evaluation import WalkForwardEvaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="limit events for a smoke run")
    parser.add_argument("--start", type=int, default=0, help="zero-based row offset")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--tag", default="v5")
    parser.add_argument("--llm-mode", default="none", choices=["none", "heuristic", "llm"])
    parser.add_argument("--llm-seed", type=int, default=0)
    args = parser.parse_args()

    evaluator = WalkForwardEvaluator(llm_mode=args.llm_mode, llm_seed=args.llm_seed)
    df = evaluator.run(limit=args.limit or None, start=args.start,
                       log_every=args.log_every)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"all_experiment_results_finmas_{args.tag}_walk_forward.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    committed = df[~df["abstain"]]
    print(f"rows={len(df)} committed={len(committed)}")
    if len(df):
        print(f"full direction accuracy={df['dir_correct'].mean():.4f}")
    if len(committed):
        print(f"committed direction accuracy={committed['committed_dir_correct'].mean():.4f}")
    print(f"output={out}")


if __name__ == "__main__":
    main()
