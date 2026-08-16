import pandas as pd

from finmas.sim.causal_graph import CausalGraph
from finmas.sim.simulator import MarketSimulator


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

