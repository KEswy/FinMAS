"""Walk-forward weight/threshold selection for two probability result files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from finmas.ensemble import walk_forward_ensemble
from finmas.reporting import (
    committed_paired_bootstrap,
    join_event_type,
    paired_bootstrap,
    selective_curve,
    summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--abstain-event-types", default="")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--legacy-csv",
        default="data/processed/all_experiment_results_deepseek-v4-flash_wf_deepseek_flash_321_gpu_20260815_204839.csv",
    )
    args = parser.parse_args()
    abstain_types = (
        [x.strip() for x in args.abstain_event_types.split(",") if x.strip()]
        or None
    )

    df = walk_forward_ensemble(
        pd.read_csv(args.a),
        pd.read_csv(args.b),
        abstain_event_types=abstain_types,
    )
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    legacy = join_event_type(pd.read_csv(args.legacy_csv))
    report = {
        "summary": summary(join_event_type(df)),
        "selective_curve": selective_curve(join_event_type(df)),
        "paired_bootstrap_full": paired_bootstrap(legacy, df, n_boot=2000),
        "paired_bootstrap_committed": committed_paired_bootstrap(
            legacy, df, n_boot=2000
        ),
        "abstain_event_types": abstain_types,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

