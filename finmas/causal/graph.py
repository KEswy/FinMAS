"""Temporal causal graph with NetworkX and SQLite snapshots."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import pandas as pd

from .schemas import CausalEdge, CausalPath


class TemporalCausalGraph:
    def __init__(self, checkpoint_path: str = "data/eval/causal_graph.sqlite",
                 decay_half_life_days: float = 30.0) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.decay_half_life_days = float(decay_half_life_days)
        self.graph = nx.DiGraph()
        self._load()

    def _load(self) -> None:
        if not self.checkpoint_path.exists():
            return
        conn = sqlite3.connect(str(self.checkpoint_path))
        try:
            rows = conn.execute(
                "SELECT source, target, relation, time, weight, confidence, evidence FROM edges"
            ).fetchall()
            for source, target, relation, time, weight, confidence, evidence in rows:
                self.graph.add_edge(
                    source,
                    target,
                    relation=relation,
                    time=time,
                    weight=float(weight),
                    confidence=float(confidence),
                    evidence=json.loads(evidence or "[]"),
                )
        finally:
            conn.close()

    def _save(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.checkpoint_path))
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS edges (
                    source TEXT,
                    target TEXT,
                    relation TEXT,
                    time TEXT,
                    weight REAL,
                    confidence REAL,
                    evidence TEXT
                )
                """
            )
            conn.execute("DELETE FROM edges")
            for source, target, data in self.graph.edges(data=True):
                conn.execute(
                    "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        source,
                        target,
                        str(data.get("relation", "affects")),
                        str(data.get("time", "")),
                        float(data.get("weight", 1.0)),
                        float(data.get("confidence", 1.0)),
                        json.dumps(data.get("evidence", []), ensure_ascii=False),
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def _effective_weight(self, data: dict, as_of: str) -> float:
        try:
            age_days = (pd.Timestamp(as_of) - pd.Timestamp(data.get("time", as_of))).days
        except Exception:
            age_days = 0.0
        decay = math.exp(-math.log(2) * max(age_days, 0) / self.decay_half_life_days)
        return float(data.get("weight", 1.0)) * decay * float(data.get("confidence", 1.0))

    def add_edge(self, edge: CausalEdge) -> None:
        self.graph.add_edge(
            edge.source,
            edge.target,
            relation=edge.relation,
            time=edge.time,
            weight=edge.weight,
            confidence=edge.confidence,
            evidence=edge.evidence_ids,
        )
        self._save()

    def edges(self, as_of: Optional[str] = None) -> List[CausalEdge]:
        out = []
        for source, target, data in self.graph.edges(data=True):
            edge = CausalEdge(
                source=source,
                target=target,
                relation=str(data.get("relation", "affects")),
                time=str(data.get("time", "")),
                weight=float(data.get("weight", 1.0)),
                confidence=float(data.get("confidence", 1.0)),
                evidence_ids=list(data.get("evidence", [])),
            )
            out.append(edge)
        return out

    def temporal_pagerank(
        self,
        personalization: Optional[Dict[str, float]] = None,
        as_of: Optional[str] = None,
    ) -> Dict[str, float]:
        working = nx.DiGraph()
        for source, target, data in self.graph.edges(data=True):
            weight = self._effective_weight(data, as_of or data.get("time", "1970-01-01"))
            if weight > 0:
                working.add_edge(source, target, weight=weight)
        if not working:
            return {}
        return nx.pagerank(
            working,
            personalization=personalization,
            weight="weight",
            max_iter=200,
        )

    def shortest_causal_paths(
        self,
        source: str,
        target: str,
        top_k: int = 5,
        as_of: Optional[str] = None,
    ) -> List[CausalPath]:
        working = nx.DiGraph()
        for s, t, data in self.graph.edges(data=True):
            weight = max(1e-6, 1.0 - min(self._effective_weight(data, as_of or data["time"]), 0.999))
            working.add_edge(s, t, weight=weight)
        if source not in working or target not in working:
            return []
        paths = []
        try:
            import itertools

            for path in itertools.islice(
                nx.shortest_simple_paths(working, source, target, weight="weight"),
                top_k,
            ):
                edges = []
                conf = 1.0
                for a, b in zip(path[:-1], path[1:]):
                    data = self.graph[a][b]
                    edges.append(
                        CausalEdge(
                            source=a,
                            target=b,
                            relation=data.get("relation", "affects"),
                            time=data.get("time", ""),
                            weight=data.get("weight", 1.0),
                            confidence=data.get("confidence", 1.0),
                            evidence_ids=data.get("evidence", []),
                        )
                    )
                    conf *= data.get("confidence", 1.0)
                paths.append(
                    CausalPath(
                        nodes=path,
                        edges=edges,
                        weight=float(len(edges)),
                        confidence=float(conf),
                        evidence_ids=sorted(
                            {eid for e in edges for eid in e.evidence_ids}
                        ),
                    )
                )
        except Exception:
            return []
        return paths

    def to_dict(self) -> Dict[str, object]:
        return {
            "nodes": list(self.graph.nodes()),
            "edges": [e.to_dict() for e in self.edges()],
        }
