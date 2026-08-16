"""Simulation report helpers."""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import SimulationState


def save_json(state: SimulationState, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def to_html(state: SimulationState) -> str:
    rows = []
    for entry in state.timeline:
        rows.append(
            "<tr>"
            f"<td>{entry.date}</td>"
            f"<td>{entry.tick.market_return:+.4f}</td>"
            f"<td>{len(entry.opinions)}</td>"
            f"<td>{entry.narrative}</td>"
            "</tr>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>FinMAS Simulation</title></head><body>"
        "<h1>Market Simulation</h1><table border='1'>"
        "<tr><th>Date</th><th>Market Return</th><th>Opinions</th><th>Narrative</th></tr>"
        + "".join(rows)
        + "</table></body></html>"
    )

