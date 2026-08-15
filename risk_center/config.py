"""Configuration for the risk-center analytics service."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _latest_result_csv() -> str:
    candidates = sorted(
        Path("data/processed").glob("all_experiment_results_*321*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return str(candidates[0])
    return "data/processed/all_experiment_results_deepseek-v4-flash_wf_deepseek_flash_321_gpu_20260815_204839.csv"


@dataclass
class RiskCenterConfig:
    result_csv: str = field(
        default_factory=lambda: os.environ.get("RISK_CENTER_RESULT_CSV", _latest_result_csv())
    )
    car_csv: str = "data/processed/car_results_expanded.csv"
    industry_daily_csv: str = "data/raw/industry_daily.csv"
    market_daily_csv: str = "data/raw/hs300.csv"
    lpr_csv: str = "data/raw/lpr.csv"
    margin_csv: str = "data/raw/margin_daily.csv"
    fundamentals_csv: str = "data/raw/industry_fundamentals.csv"
    output_dir: str = "data/risk_center"
    horizons: tuple[int, ...] = (1, 5, 20)
    alphas: tuple[float, ...] = (0.05, 0.01)
    risk_methods: tuple[str, ...] = (
        "historical",
        "bootstrap",
        "parametric",
        "ewma",
        "evt",
        "model_interval",
    )
    beta_window: int = 60
    n_bootstrap: int = 1000
    seed: int = 42
    weights_path: Optional[str] = None
    limits: Dict[str, float] = field(
        default_factory=lambda: {
            "single_industry_weight": 0.20,
            "hhi_warning": 0.10,
            "breach_warning_multiple": 2.0,
            "portfolio_var_budget": 0.10,
        }
    )
    stress_scenarios: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.horizons = tuple(int(h) for h in self.horizons)
        self.alphas = tuple(float(a) for a in self.alphas)
        self.risk_methods = tuple(str(m) for m in self.risk_methods)
        self.stress_scenarios = self.stress_scenarios or {
            "historical": [
                {"name": "2020_covid", "start_date": "2020-02-03", "end_date": "2020-03-23"},
                {"name": "2024_sep_policy", "start_date": "2024-09-24", "end_date": "2024-10-08"},
                {"name": "2025_tariff", "start_date": "2025-04-02", "end_date": "2025-04-09"},
            ],
            "synthetic": [
                {"name": "market_minus_5pct", "market_shock": -0.05},
                {"name": "market_minus_10pct", "market_shock": -0.10},
                {"name": "vol_up", "market_shock": -0.03, "vol_multiplier": 2.0},
            ],
        }


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        out[key] = value
    return out


def load_config(path: Optional[str | Path] = None) -> RiskCenterConfig:
    config = RiskCenterConfig()
    if not path:
        return config

    path = Path(path)
    if not path.exists():
        return config

    raw: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    config.result_csv = str(raw.get("result_csv", config.result_csv))
    config.car_csv = str(raw.get("car_csv", config.car_csv))
    config.industry_daily_csv = str(raw.get("industry_daily_csv", config.industry_daily_csv))
    config.market_daily_csv = str(raw.get("market_daily_csv", config.market_daily_csv))
    config.lpr_csv = str(raw.get("lpr_csv", config.lpr_csv))
    config.margin_csv = str(raw.get("margin_csv", config.margin_csv))
    config.fundamentals_csv = str(raw.get("fundamentals_csv", config.fundamentals_csv))
    config.output_dir = str(raw.get("output_dir", config.output_dir))
    config.horizons = tuple(int(x) for x in raw.get("horizons", config.horizons))
    config.alphas = tuple(float(x) for x in raw.get("alphas", config.alphas))
    config.risk_methods = tuple(str(x) for x in raw.get("risk_methods", config.risk_methods))
    config.beta_window = int(raw.get("beta_window", config.beta_window))
    config.n_bootstrap = int(raw.get("n_bootstrap", config.n_bootstrap))
    config.seed = int(raw.get("seed", config.seed))
    config.weights_path = raw.get("weights_path", config.weights_path)
    config.limits = _merge_dict(config.limits, raw.get("limits", {}))
    config.stress_scenarios = _merge_dict(config.stress_scenarios, raw.get("stress_scenarios", {}))
    return config
