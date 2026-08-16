"""Optional LangGraph orchestration shell with local fallback."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from ..pipeline import FinMASPipeline
from ..schemas import EventInput
from .nodes import (
    debate_node,
    fusion_node,
    mechanism_node,
    retrieval_node,
    risk_node,
    time_series_node,
)
from .state import GraphState


class FinMASGraph:
    def __init__(
        self,
        pipeline: Optional[FinMASPipeline] = None,
        use_langgraph: bool = False,
        debate_rounds: int = 2,
        checkpoint_path: str = "data/eval/graph_checkpoint.sqlite",
    ) -> None:
        self.pipeline = pipeline or FinMASPipeline()
        self.use_langgraph = use_langgraph
        self.debate_rounds = int(debate_rounds)
        self.checkpoint_path = Path(checkpoint_path)
        self._graph = self._build_langgraph() if use_langgraph else None

    def _build_langgraph(self):
        try:
            from langgraph.graph import StateGraph
        except Exception:
            return None

        graph = StateGraph(dict)
        graph.add_node("retrieval", lambda s: retrieval_node(s, self.pipeline))
        graph.add_node("mechanism", lambda s: mechanism_node(s, self.pipeline))
        graph.add_node(
            "debate",
            lambda s: debate_node(s, self.pipeline, self.debate_rounds),
        )
        graph.add_node("time_series", lambda s: time_series_node(s, self.pipeline))
        graph.add_node("fusion", lambda s: fusion_node(s, self.pipeline))
        graph.add_node("risk", lambda s: risk_node(s, self.pipeline))
        graph.set_entry_point("retrieval")
        graph.add_edge("retrieval", "mechanism")
        graph.add_edge("mechanism", "debate")
        graph.add_edge("debate", "time_series")
        graph.add_edge("time_series", "fusion")
        graph.add_edge("fusion", "risk")
        return graph.compile()

    def run(self, event: EventInput) -> GraphState:
        state = GraphState(event=event)
        if self._graph is not None:
            state = self._graph.invoke(state)
            self._save_checkpoint(state)
            return state

        retrieval_node(state, self.pipeline)
        mechanism_node(state, self.pipeline)
        debate_node(state, self.pipeline, self.debate_rounds)
        time_series_node(state, self.pipeline)
        fusion_node(state, self.pipeline)
        risk_node(state, self.pipeline)
        self._save_checkpoint(state)
        return state

    def _save_checkpoint(self, state: GraphState) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.checkpoint_path))
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_checkpoints (
                    event_date TEXT,
                    industry_code TEXT,
                    state_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO graph_checkpoints
                    (event_date, industry_code, state_json)
                VALUES (?, ?, ?)
                """,
                (
                    state.event.event_date,
                    state.event.industry_code,
                    json.dumps(state.to_dict(), ensure_ascii=False, default=str),
                ),
            )
            conn.commit()
        finally:
            conn.close()

