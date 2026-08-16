import pandas as pd

from finmas.sim.causal_graph import CausalGraph
from finmas.sim.data_source import MarketDataSource
from finmas.sim.schemas import MarketTick
from finmas.sim.simulator import MarketSimulator
from finmas.quality.explanation_quality import compute_explanation_metrics


def test_market_simulator_runs_and_produces_timeline(tmp_path):
    market = pd.read_csv("data/raw/hs300.csv")
    market["日期"] = pd.to_datetime(market["日期"])
    dates = market["日期"].head(3).dt.strftime("%Y-%m-%d").tolist()
    sim = MarketSimulator()
    state = sim.run(dates)
    assert len(state.timeline) == len(dates)
    assert state.timeline[0].opinions
    assert state.timeline[0].causal_paths


def test_counterfactual_result():
    sim = MarketSimulator()
    result = sim.counterfactual(
        "2024-01-02",
        {"market_return": -0.03, "triggered_event": "合成利空事件"},
    )
    assert result.original_narrative
    assert result.counterfactual_narrative


def test_causal_graph_persists(tmp_path):
    path = tmp_path / "graph.sqlite"
    graph = CausalGraph(str(path))
    graph.add_path(
        __import__("finmas.sim.schemas", fromlist=["CausalPath"]).CausalPath(
            "event", "market", "causes", 0.8, ["e1"]
        )
    )
    reloaded = CausalGraph(str(path))
    assert reloaded.paths()


def test_event_detection_flags_large_market_move():
    ds = MarketDataSource()
    tick = MarketTick(
        date="2024-01-02",
        market_return=-0.03,
        industry_returns={"801780": -0.05},
        capital_flow={"margin_change": -0.02},
    )
    detected = ds.detect_event(tick)
    assert detected.triggered_event


def test_explanation_quality_report_from_state():
    market = pd.read_csv("data/raw/hs300.csv")
    market["日期"] = pd.to_datetime(market["日期"])
    dates = market["日期"].head(5).dt.strftime("%Y-%m-%d").tolist()
    state = MarketSimulator().run(dates)
    report = compute_explanation_metrics(state)
    assert report.module == "explanation"
    assert report.metrics
