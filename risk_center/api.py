"""FastAPI interface for the risk-center analytics service."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .config import load_config
from .report import build_full_report


app = FastAPI(title="Financial Risk Center", version="0.1.0")


class WeightRequest(BaseModel):
    weights: Optional[Dict[str, Any]] = None
    result_csv: Optional[str] = None
    config_path: Optional[str] = None


class StressRequest(BaseModel):
    weights: Optional[Dict[str, Any]] = None
    scenarios: Optional[Dict[str, Any]] = None
    result_csv: Optional[str] = None
    config_path: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "financial-risk-center"}


@app.get("/reports")
def list_reports() -> list:
    config = load_config()
    output = Path(config.output_dir)
    if not output.exists():
        return []
    return [p.name for p in sorted(output.glob("*"))]


@app.post("/risk/report")
def risk_report(request: WeightRequest) -> dict:
    config = load_config(request.config_path) if request.config_path else load_config()
    report = build_full_report(
        result_csv=request.result_csv,
        config=config,
        weights=request.weights,
    )
    return report


@app.post("/risk/stress")
def stress_report(request: StressRequest) -> dict:
    config = load_config(request.config_path) if request.config_path else load_config()
    report = build_full_report(
        result_csv=request.result_csv,
        config=config,
        weights=request.weights,
        scenarios=request.scenarios,
    )
    return report["stress"]


@app.post("/attribution/report")
def attribution_report(request: WeightRequest) -> dict:
    config = load_config(request.config_path) if request.config_path else load_config()
    report = build_full_report(
        result_csv=request.result_csv,
        config=config,
        weights=request.weights,
    )
    return report["attribution"]


@app.post("/config/weights")
def save_weights(payload: Dict[str, Any]) -> dict:
    config = load_config()
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "external_weights.json"
    path.write_text(
        __import__("json").dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"saved": str(path), "weights": payload}


@app.post("/config/stress")
def save_stress_scenarios(payload: Dict[str, Any]) -> dict:
    config = load_config()
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "stress_scenarios.json"
    path.write_text(
        __import__("json").dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"saved": str(path), "scenarios": payload}
