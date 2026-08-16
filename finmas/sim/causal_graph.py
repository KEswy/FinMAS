"""Causal graph for market simulation, backed by NetworkX + SQLite."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import networkx as nx

from .schemas import CausalPath


class CausalGraph:
    def __init__(self, checkpoint_path: str = "data/eval/sim_graph.sqlite") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.graph = nx.DiGraph()
        self._load()

    def _load(self) -> None:
        if not self.checkpoint_path.exists():
            return
        conn = sqlite3.connect(str(self.checkpoint_path))
        try:
            rows = conn.execute(
                "SELECT source, target, relation, weight, evidence FROM edges"
            ).fetchall()
            for source, target, relation, weight, evidence in rows:
                self.graph.add_edge(
                    source,
                    target,
                    relation=relation,
                    weight=float(weight),
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
                    weight REAL,
                    evidence TEXT
                )
                """
            )
            conn.execute("DELETE FROM edges")
            for source, target, data in self.graph.edges(data=True):
                conn.execute(
                    "INSERT INTO edges VALUES (?, ?, ?, ?, ?)",
                    (
                        source,
                        target,
                        str(data.get("relation", "affects")),
                        float(data.get("weight", 1.0)),
                        json.dumps(data.get("evidence", []), ensure_ascii=False),
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def add_path(self, path: CausalPath) -> None:
        self.graph.add_edge(
            path.source,
            path.target,
            relation=path.relation,
            weight=path.weight,
            evidence=path.evidence_ids,
        )
        self._save()

    def paths(self) -> List[CausalPath]:
        out = []
        for source, target, data in self.graph.edges(data=True):
            out.append(
                CausalPath(
                    source=source,
                    target=target,
                    relation=str(data.get("relation", "affects")),
                    weight=float(data.get("weight", 1.0)),
                    evidence_ids=list(data.get("evidence", [])),
                )
            )
        return out

    def to_dict(self) -> Dict[str, object]:
        return {
            "nodes": list(self.graph.nodes()),
            "edges": [p.to_dict() for p in self.paths()],
        }

