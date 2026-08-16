"""FastAPI endpoints for FinMAS v5."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
import json as _json

from ..pipeline import FinMASPipeline
from ..schemas import EventInput
from ..sim.simulator import MarketSimulator
from ..causal.graph import TemporalCausalGraph
from ..causal.explanation import CausalExplainer
from ..causal.extractor import CausalExtractor


class EventRequest(BaseModel):
    event_date: str
    event_text: str
    event_type: str
    industry_code: str
    event_id: str = ""
    external_features: Dict[str, Any] = Field(default_factory=dict)
    use_llm: bool = True


app = FastAPI(title="FinMAS v5", version="5.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "finmas-v5"}


@app.post("/predict")
def predict_direction(request: EventRequest) -> dict:
    event = EventInput(
        event_date=request.event_date,
        event_text=request.event_text,
        event_type=request.event_type,
        industry_code=request.industry_code,
        event_id=request.event_id,
        external_features=request.external_features,
    )
    pipeline = FinMASPipeline(use_llm=request.use_llm)
    decision = pipeline.predict(event)
    return decision.to_dict()


@app.post("/sim/run")
def sim_run(request: dict) -> dict:
    dates = request.get("dates")
    if not dates:
        import pandas as pd

        market = pd.read_csv("data/raw/hs300.csv")
        market["日期"] = pd.to_datetime(market["日期"])
        dates = market["日期"].head(30).dt.strftime("%Y-%m-%d").tolist()
    state = MarketSimulator().run(dates)
    return state.to_dict()


@app.get("/sim/timeline")
def sim_timeline() -> dict:
    path = Path("data/eval/sim_state.json")
    if not path.exists():
        return {"timeline": []}
    state = _json.loads(path.read_text(encoding="utf-8"))
    return {"timeline": state.get("timeline", [])}


@app.get("/sim/causal-graph")
def sim_causal_graph() -> dict:
    path = Path("data/eval/sim_state.json")
    if not path.exists():
        return {"nodes": [], "edges": []}
    state = _json.loads(path.read_text(encoding="utf-8"))
    edges = []
    nodes = set()
    for entry in state.get("timeline", []):
        for p in entry.get("causal_paths", []):
            nodes.add(p["source"])
            nodes.add(p["target"])
            edges.append(p)
    return {"nodes": sorted(nodes), "edges": edges}


@app.get("/sim/report")
def sim_report() -> dict:
    path = Path("data/eval/sim_state.json")
    if not path.exists():
        return {}
    return _json.loads(path.read_text(encoding="utf-8"))


@app.post("/causal/extract")
def causal_extract(request: dict) -> dict:
    edges = CausalExtractor(use_llm=bool(request.get("llm", False))).extract(
        request.get("event_text", ""),
        request.get("event_date", ""),
        request.get("industries", []),
    )
    return {"edges": [e.to_dict() for e in edges]}


@app.get("/causal/graph")
def causal_graph() -> dict:
    from ..sim.schemas import SimulationState
    from ..causal.schemas import CausalEdge

    path = Path("data/eval/sim_state.json")
    graph = TemporalCausalGraph()
    if path.exists():
        state = SimulationState.from_dict(_json.loads(path.read_text(encoding="utf-8")))
        for entry in state.timeline:
            for path in entry.causal_paths:
                graph.add_edge(
                    CausalEdge(
                        source=path.source,
                        target=path.target,
                        relation=path.relation,
                        time=entry.date,
                        weight=path.weight,
                        confidence=0.7,
                        evidence_ids=path.evidence_ids,
                    )
                )
    return graph.to_dict()


@app.post("/causal/explain")
def causal_explain(request: dict) -> dict:
    graph = TemporalCausalGraph()
    explainer = CausalExplainer(graph)
    return explainer.explain(
        request.get("event", "event"),
        request.get("target", "market"),
    ).to_dict()


@app.post("/causal/counterfactual")
def causal_counterfactual(request: dict) -> dict:
    graph = TemporalCausalGraph()
    explainer = CausalExplainer(graph)
    return explainer.counterfactual(
        request.get("event", "event"),
        request.get("target", "market"),
        request.get("perturbation", {}),
    )
