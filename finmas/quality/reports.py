"""Generate module-level quality reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .schemas import QualityMetric, QualityReport


def llm_quality_report(cache_path: str = "data/cache/llm_cache.json") -> QualityReport:
    path = Path(cache_path)
    report = QualityReport(module="llm", n_samples=0)
    if not path.exists():
        report.notes.append("no LLM cache")
        report.metrics.append(QualityMetric("json_parse_rate", 0.0, False, 0.9, ">="))
        return report.finalize()
    raw = json.loads(path.read_text(encoding="utf-8"))
    contents = list(raw.values())
    report.n_samples = len(contents)
    parse_ok = 0
    schema_ok = 0
    factor_ok = 0
    for value in contents:
        text = str(value.get("content", ""))
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            parse_ok += 1
        if isinstance(parsed, dict) and isinstance(parsed.get("chain"), list):
            schema_ok += 1
            if parsed["chain"]:
                factor_ok += 1
    n = max(report.n_samples, 1)
    report.metrics.extend(
        [
            QualityMetric("json_parse_rate", parse_ok / n, parse_ok / n >= 0.90, 0.90, ">="),
            QualityMetric("schema_valid_rate", schema_ok / n, schema_ok / n >= 0.85, 0.85, ">="),
            QualityMetric("factor_coverage", factor_ok / n, factor_ok / n >= 0.80, 0.80, ">="),
        ]
    )
    return report.finalize()


def fusion_quality_report(
    result_csv: str,
    min_coverage: float = 0.40,
) -> QualityReport:
    path = Path(result_csv)
    report = QualityReport(module="fusion", n_samples=0)
    if not path.exists():
        report.metrics.append(QualityMetric("selective_accuracy", 0.0, False, 0.57, ">="))
        return report.finalize()
    df = pd.read_csv(path)
    report.n_samples = len(df)
    if "committed_dir_correct" not in df.columns or "abstain" not in df.columns:
        report.metrics.append(QualityMetric("selective_accuracy", 0.0, False, 0.57, ">="))
        return report.finalize()
    committed = df[~df["abstain"]]
    if len(committed) < max(5, len(df) * min_coverage):
        value = 0.0
    else:
        value = float(committed["committed_dir_correct"].mean())
    full = float(df["dir_correct"].mean()) if "dir_correct" in df.columns else 0.0
    report.metrics.extend(
        [
            QualityMetric("full_accuracy", full, full >= 0.50, 0.50, ">="),
            QualityMetric("selective_accuracy", value, value >= 0.57, 0.57, ">="),
            QualityMetric(
                "coverage",
                float(len(committed) / max(len(df), 1)),
                len(committed) / max(len(df), 1) >= min_coverage,
                min_coverage,
                ">=",
            ),
        ]
    )
    return report.finalize()


def risk_quality_report(risk_json: str) -> QualityReport:
    report = QualityReport(module="risk", n_samples=0)
    path = Path(risk_json)
    if not path.exists():
        report.metrics.append(QualityMetric("kupiec_pass", 0.0, False, 1.0, ">="))
        return report.finalize()
    raw = json.loads(path.read_text(encoding="utf-8"))
    horizons = raw.get("horizons", {})
    report.n_samples = len(horizons)
    passes = 0
    for horizon, value in horizons.items():
        v5 = value.get("v5", {})
        p = float(v5.get("kupiec_p", 0.0))
        if p >= 0.05:
            passes += 1
    rate = passes / max(len(horizons), 1)
    report.metrics.append(
        QualityMetric("kupiec_pass_rate", rate, rate >= 0.66, 0.66, ">=")
    )
    return report.finalize()


def generate_all_quality_reports(
    result_csv: str,
    risk_json: str,
    output_dir: str = "data/eval",
) -> Dict[str, Dict[str, Any]]:
    reports = {
        "rag": None,
        "llm": llm_quality_report().to_dict(),
        "fusion": fusion_quality_report(result_csv).to_dict(),
        "risk": risk_quality_report(risk_json).to_dict(),
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for module, payload in reports.items():
        if payload:
            (out / f"quality_{module}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    return {k: v for k, v in reports.items() if v}
