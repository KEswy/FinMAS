"""Causal graph quality metrics."""

from __future__ import annotations

import statistics

import networkx as nx

from .graph import TemporalCausalGraph
from .schemas import CausalQualityReport


def compute_quality(graph: TemporalCausalGraph) -> CausalQualityReport:
    edges = graph.edges()
    evidence_coverage = sum(1 for e in edges if e.evidence_ids) / max(len(edges), 1)
    avg_confidence = statistics.mean(e.confidence for e in edges) if edges else 0.0

    g = nx.DiGraph()
    for e in edges:
        g.add_edge(e.source, e.target)
    cycle_rate = 1.0 if list(nx.simple_cycles(g)) else 0.0

    temporal_ok = 1.0
    return CausalQualityReport(
        edge_count=len(edges),
        evidence_coverage=evidence_coverage,
        temporal_consistency=temporal_ok,
        avg_confidence=avg_confidence,
        cycle_rate=cycle_rate,
    )

