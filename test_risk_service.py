"""Tests for the offline risk-report service."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.risk_service import (
    RiskReportConfig,
    aggregate_risk,
    build_risk_report,
    load_weights,
)


def _row(code, realized, var, mdd=0.01):
    return {
        "event_date": "2024-01-02",
        "event_name": f"event-{code}",
        "industry_code": str(code),
        "dispersion": 0.1,
        "realized_car_T1": realized,
        "realized_mdd_T1": mdd,
        "realized_downside_T1": max(-realized, 0.0),
        "pred_var_T1": var,
        "pred_mdd_T1": mdd,
    }


def test_aggregate_risk_equal_weight_and_industry_grouping():
    rows = [
        _row("801780", -0.03, -0.05),
        _row("801780", -0.01, -0.04),
        _row("801750", 0.02, -0.03),
    ]
    report = aggregate_risk(rows, weights=None, horizons=(1,), n_bootstrap=10)
    portfolio = report["portfolio_risk"]["1"]
    assert abs(portfolio["hhi"] - 1.0 / 3.0) < 1e-6
    assert abs(portfolio["effective_n"] - 3.0) < 1e-9
    assert abs(portfolio["realized_car"] - np.mean([-0.03, -0.01, 0.02])) < 1e-6
    industries = report["industry_risk"]["1"]
    assert {x["industry_code"] for x in industries} == {"801780", "801750"}
    assert report["attribution"]["1"]["top_loss_events"]


def test_industry_weight_normalization():
    rows = [_row("801780", -0.03, -0.05), _row("801750", 0.02, -0.03)]
    weights = {"by_industry": {"801780": 2.0, "801750": 1.0}}
    report = aggregate_risk(rows, weights=weights, horizons=(1,), n_bootstrap=10)
    industry_weights = {
        x["industry_code"]: x["weight"] for x in report["industry_risk"]["1"]
    }
    assert abs(industry_weights["801780"] - 2.0 / 3.0) < 1e-6
    assert abs(industry_weights["801750"] - 1.0 / 3.0) < 1e-6


def test_build_report_legacy_columns_and_realized_risk():
    dates = pd.bdate_range("2024-01-02", periods=40)
    industry_rows = []
    market_rows = []
    for code in ("801780", "801750"):
        close = 100.0
        for i, d in enumerate(dates):
            close *= 1.0 + (0.001 if code == "801780" else -0.0005)
            industry_rows.append(
                {"日期": d, "收盘": close, "industry_code": code}
            )
    close = 100.0
    for d in dates:
        close *= 1.0 + 0.0002
        market_rows.append({"日期": d, "收盘": close})

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        industry_path = tmp / "industry_daily.csv"
        market_path = tmp / "hs300.csv"
        csv_path = tmp / "results.csv"
        pd.DataFrame(industry_rows).to_csv(industry_path, index=False)
        pd.DataFrame(market_rows).to_csv(market_path, index=False)
        pd.DataFrame(
            [
                {
                    "event_date": "2024-01-02",
                    "event_name": "synthetic",
                    "industry_code": "801780",
                    "event_VaR": -0.05,
                    "expected_MDD": 0.02,
                    "tail_prob": 0.05,
                    "risk_lo": -0.06,
                    "risk_hi": 0.04,
                    "dispersion": 0.1,
                }
            ]
        ).to_csv(csv_path, index=False)

        cfg = RiskReportConfig(
            horizons=(1, 5, 20),
            industry_daily_csv=str(industry_path),
            market_daily_csv=str(market_path),
            output_dir=str(tmp / "out"),
            output_csv=False,
            n_bootstrap=10,
        )
        report = build_risk_report(str(csv_path), cfg)
        assert set(report["calibration"].keys()) == {"1", "5", "20"}
        assert report["calibration"]["5"] is not None
        assert report["calibration"]["1"] is None
        assert report["calibration"]["20"] is None
        assert set(report["event_risk"].keys()) == {"1", "5", "20"}
        assert report["event_risk"]["5"][0]["realized_car"] is not None
        out = next((tmp / "out").glob("*_risk_report.json"))
        assert json.loads(out.read_text(encoding="utf-8"))["meta"]["n_events"] == 1


def test_load_weights(tmp_path):
    p = tmp_path / "weights.json"
    p.write_text(
        json.dumps({"by_industry": {"801780": 0.7}}),
        encoding="utf-8",
    )
    assert load_weights(str(p)) == {"by_industry": {"801780": 0.7}}


if __name__ == "__main__":
    test_aggregate_risk_equal_weight_and_industry_grouping()
    test_industry_weight_normalization()
    test_build_report_legacy_columns_and_realized_risk()
    test_load_weights(Path(tempfile.gettempdir()))
    print("ALL RISK SERVICE TESTS PASSED")
