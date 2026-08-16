"""Summarize multi-seed v5 results as mean/std."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    full_accs = []
    committed_accs = []
    coverages = []
    details = []
    for path in args.inputs.split(","):
        path = path.strip()
        if not path:
            continue
        df = pd.read_csv(path)
        full = float(df["dir_correct"].mean())
        committed = df[~df["abstain"]] if "abstain" in df else df
        committed_acc = (
            float(committed["committed_dir_correct"].mean())
            if "committed_dir_correct" in committed
            else float("nan")
        )
        coverage = float(len(committed) / max(len(df), 1))
        full_accs.append(full)
        committed_accs.append(committed_acc)
        coverages.append(coverage)
        details.append(
            {
                "file": path,
                "full_accuracy": full,
                "committed_accuracy": committed_acc,
                "coverage": coverage,
            }
        )

    report = {
        "details": details,
        "full_mean": float(np.mean(full_accs)) if full_accs else float("nan"),
        "full_std": float(np.std(full_accs)) if full_accs else float("nan"),
        "committed_mean": float(np.mean(committed_accs)) if committed_accs else float("nan"),
        "committed_std": float(np.std(committed_accs)) if committed_accs else float("nan"),
        "coverage_mean": float(np.mean(coverages)) if coverages else float("nan"),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

