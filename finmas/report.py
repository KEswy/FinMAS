"""Consolidated Markdown report builder for FinMAS v5 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .reporting import join_event_type, selective_curve, summary


def build_markdown_report(
    result_csv: str,
    risk_json: Optional[str] = None,
    portfolio_json: Optional[str] = None,
    legacy_csv: str = "data/processed/all_experiment_results_deepseek-v4-flash_wf_deepseek_flash_321_gpu_20260815_204839.csv",
) -> str:
    df = join_event_type(pd.read_csv(result_csv))
    s = summary(df)
    curve = selective_curve(df)

    lines = ["# FinMAS v5 Report", ""]
    lines.append(f"Result CSV: `{result_csv}`")
    lines.append(f"Rows: {s['full']['n']}")
    lines.append(
        f"Full accuracy: {s['full']['accuracy']:.2%}  "
        f"Committed accuracy: {s['selective']['accuracy']:.2%}  "
        f"Coverage: {s['selective']['coverage']:.2%}"
    )
    lines.append("")
    lines.append("## Selective curve")
    lines.append("| Coverage | Accuracy | Min confidence |")
    lines.append("|---:|---:|---:|")
    for row in curve[:5]:
        lines.append(
            f"| {row['coverage']:.2%} | {row['accuracy']:.2%} | "
            f"{row['min_confidence']:.3f} |"
        )

    if risk_json and Path(risk_json).exists():
        risk = json.loads(Path(risk_json).read_text(encoding="utf-8"))
        lines.append("")
        lines.append("## Event-level risk")
        lines.append("| Horizon | v5 breach | legacy breach | v5 Kupiec p |")
        lines.append("|---:|---:|---:|---:|")
        for horizon, value in risk.get("horizons", {}).items():
            v5 = value.get("v5", {})
            legacy = value.get("legacy", {})
            lines.append(
                f"| {horizon} | {value.get('v5_breach_rate', 0):.2%} | "
                f"{value.get('legacy_breach_rate', 0):.2%} | "
                f"{v5.get('kupiec_p', float('nan')):.3f} |"
            )

    if portfolio_json and Path(portfolio_json).exists():
        portfolio = json.loads(Path(portfolio_json).read_text(encoding="utf-8"))
        lines.append("")
        lines.append("## Portfolio risk")
        lines.append("| Horizon | breach | Kupiec p | pinball |")
        lines.append("|---:|---:|---:|---:|")
        for horizon, value in portfolio.get("horizons", {}).items():
            bt = value.get("backtest", {})
            lines.append(
                f"| {horizon} | {bt.get('breach_rate', 0):.2%} | "
                f"{bt.get('kupiec_p', float('nan')):.3f} | "
                f"{bt.get('pinball', float('nan')):.5f} |"
            )

    return "\n".join(lines)

