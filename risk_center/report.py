"""Full report orchestration, persistence, and HTML visualization."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from evaluation.risk_service import load_weights

from .attribution import (
    aggregate_return_attribution,
    run_return_attribution,
    run_risk_attribution,
)
from .config import RiskCenterConfig, load_config
from .data_loader import RiskCenterData
from .portfolio_risk import aggregate_risk, enrich_rows
from .stress_test import run_stress_tests


def _accuracy_by_event_type(data: RiskCenterData) -> dict:
    frame = data.experiment_rows()
    if "event_type" not in frame.columns or frame["event_type"].isna().all():
        return {}
    grouped = (
        frame.groupby("event_type")
        .agg(n=("dir_correct", "size"), accuracy=("dir_correct", "mean"))
        .sort_values("accuracy")
    )
    return grouped.reset_index().to_dict("records")


def _save_json(report: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)


def _save_flat_csv(rows: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat = []
    for row in rows:
        item = dict(row)
        for key, value in list(item.items()):
            if isinstance(value, tuple):
                item.pop(key)
        flat.append(item)
    pd.DataFrame(flat).to_csv(out_path, index=False, encoding="utf-8-sig")


def _html_escape(text: object) -> str:
    import html
    return html.escape(str(text))


def _plotly_or_simple(report: dict, out_path: Path) -> None:
    figures = []
    try:
        import plotly.graph_objects as go

        acc = report.get("meta", {}).get("accuracy_by_event_type", [])
        if acc:
            fig = go.Figure(
                go.Bar(
                    x=[item["event_type"] for item in acc],
                    y=[item["accuracy"] for item in acc],
                    text=[f"{item['accuracy']:.2%}" for item in acc],
                    textposition="outside",
                )
            )
            fig.update_layout(title="Direction accuracy by event type", yaxis_tickformat=".0%")
            figures.append(fig)

        cal = report.get("risk", {}).get("calibration", {})
        if cal:
            horizons = list(cal.keys())
            breach = [
                float(cal[h]["breach_rate"]) if cal[h] else 0.0
                for h in horizons
            ]
            expected = [
                float(cal[h]["expected_rate"]) if cal[h] else 0.05
                for h in horizons
            ]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=horizons, y=breach, mode="lines+markers", name="breach"))
            fig.add_trace(go.Scatter(x=horizons, y=expected, mode="lines+markers", name="expected"))
            fig.update_layout(title="VaR breach rate by horizon", yaxis_tickformat=".0%")
            figures.append(fig)

        risk_attr = report.get("attribution", {}).get("risk", {}).get("components", [])
        if risk_attr:
            top = risk_attr[-15:]
            fig = go.Figure(
                go.Bar(
                    x=[item["industry_code"] for item in top],
                    y=[item["component_var"] for item in top],
                )
            )
            fig.update_layout(title="Industry component VaR", yaxis_title="VaR")
            figures.append(fig)

        stress = report.get("stress", {})
        stress_rows = []
        for scenario in stress.get("historical", []) + stress.get("synthetic", []):
            stress_rows.append((scenario["name"], scenario["portfolio_pnl"]))
        if stress_rows:
            fig = go.Figure(
                go.Bar(
                    x=[x[0] for x in stress_rows],
                    y=[x[1] for x in stress_rows],
                )
            )
            fig.update_layout(title="Stress scenario portfolio P&L", yaxis_tickformat=".1%")
            figures.append(fig)

        if figures:
            html_parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Risk Center Report</title></head><body>"]
            for fig in figures:
                html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
            html_parts.append("</body></html>")
            out_path.write_text("\n".join(html_parts), encoding="utf-8")
            return
    except Exception:
        pass

    body = ["<h1>Risk Center Report</h1>", "<pre>" + _html_escape(json.dumps(report, ensure_ascii=False, indent=2, default=str)) + "</pre>"]
    out_path.write_text("<!doctype html><html><meta charset='utf-8'><body>" + "\n".join(body) + "</body></html>", encoding="utf-8")


def build_full_report(
    result_csv: Optional[str] = None,
    config: Optional[RiskCenterConfig] = None,
    weights: Optional[dict] = None,
    scenarios: Optional[dict] = None,
) -> dict:
    config = config or load_config()
    if result_csv:
        config.result_csv = str(result_csv)

    data = RiskCenterData(config)
    rows = enrich_rows(data, config.horizons)
    if weights is None and config.weights_path:
        weights = load_weights(config.weights_path)

    risk = aggregate_risk(
        rows,
        data,
        weights=weights,
        horizons=config.horizons,
        alphas=config.alphas,
        methods=config.risk_methods,
        limits=config.limits,
        n_bootstrap=config.n_bootstrap,
        seed=config.seed,
    )
    industry_weights = risk["weights"]["by_industry"]
    stress = run_stress_tests(
        data,
        industry_weights,
        scenarios=scenarios or config.stress_scenarios,
    )
    return_attribution = run_return_attribution(
        data,
        rows,
        horizons=config.horizons,
        beta_window=config.beta_window,
    )
    return_aggregated = aggregate_return_attribution(
        return_attribution,
        group_by="event_type",
    )
    risk_attribution = run_risk_attribution(
        data,
        industry_weights,
        alpha=0.05,
        horizon=5,
    )

    stem = Path(config.result_csv).stem
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "meta": {
            "source_csv": config.result_csv,
            "stem": stem,
            "n_events": len(rows),
            "n_industries": len(industry_weights),
            "horizons": list(config.horizons),
            "alphas": list(config.alphas),
            "risk_methods": list(config.risk_methods),
            "accuracy_by_event_type": _accuracy_by_event_type(data),
            "weighting": "equal_weight" if weights is None else weights,
        },
        "risk": risk,
        "stress": stress,
        "attribution": {
            "return_by_event": return_attribution,
            "return_aggregated_by_event_type": return_aggregated,
            "risk": risk_attribution,
        },
    }

    _save_json(report, output_dir / f"{stem}_risk_center.json")
    _save_flat_csv(rows, output_dir / f"{stem}_risk_center_flat.csv")
    _plotly_or_simple(report, output_dir / f"{stem}_risk_center_report.html")
    return report
