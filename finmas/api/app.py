"""FastAPI endpoints for FinMAS v5."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ..pipeline import FinMASPipeline
from ..schemas import EventInput
from ..sim.simulator import MarketSimulator


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
    return {"message": "use /sim/run to generate a timeline"}
