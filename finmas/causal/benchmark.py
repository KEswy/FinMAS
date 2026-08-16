"""Causal graph baseline comparison."""

from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx
import pandas as pd

from ..sim.schemas import SimulationState
from .extractor import CausalExtractor
from .graph import TemporalCausalGraph
from .quality import compute_quality
from .schemas import CausalEdge, CausalQualityReport


def edges_from_sim(state: SimulationState) -> List[CausalEdge]:
    edges = []
    for entry in state.timeline:
        for path in entry.causal_paths:
            edges.append(
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
    return edges


def llm_or_rule_edges(events: List[dict], use_llm: bool = False) -> List[CausalEdge]:
    extractor = CausalExtractor(use_llm=use_llm)
    out = []
    for event in events:
        out.extend(
            extractor.extract(
                event.get("event_text", ""),
                event.get("event_date", ""),
                event.get("industries", []),
            )
        )
    return out


def rule_kg_edges(events: List[dict]) -> List[CausalEdge]:
    out = []
    for event in events:
        for industry in event.get("industries", []):
            out.append(
                CausalEdge(
                    source="event",
                    target=str(industry),
                    relation="affects",
                    time=event.get("event_date", ""),
                    weight=0.7,
                    confidence=0.7,
                    evidence_ids=["rule_kg"],
                )
            )
    return out


def ordinary_pagerank(graph: TemporalCausalGraph) -> Dict[str, float]:
    g = nx.DiGraph()
    for edge in graph.edges():
        g.add_edge(edge.source, edge.target, weight=edge.weight)
    if not g:
        return {}
    return nx.pagerank(g, weight="weight")


def compare_graphs(
    reference: List[CausalEdge],
    candidate: List[CausalEdge],
) -> Dict[str, float]:
    ref = {(e.source, e.target, e.relation) for e in reference}
    cand = {(e.source, e.target, e.relation) for e in candidate}
    tp = len(ref & cand)
    fp = len(cand - ref)
    fn = len(ref - cand)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1}


def run_benchmark(
    state: SimulationState,
    events: Optional[List[dict]] = None,
    llm_judge: bool = False,
) -> Dict[str, object]:
    graph = TemporalCausalGraph()
    for edge in edges_from_sim(state):
        graph.add_edge(edge)
    reference = edges_from_sim(state)
    if not events:
        events = []
        for entry in state.timeline:
            events.append(
                {
                    "event_date": entry.date,
                    "event_text": entry.tick.triggered_event or entry.narrative,
                    "industries": list(entry.tick.industry_returns.keys()),
                }
            )
    rule = rule_kg_edges(events)
    llm_rule = llm_or_rule_edges(events, use_llm=False)
    judge_score = _llm_judge_edges(reference) if llm_judge else 0.0
    quality = compute_quality(graph).to_dict()
    quality["llm_judge_score"] = judge_score
    return {
        "graph_quality": quality,
        "edge_counts": {
            "temporal_graph": len(reference),
            "rule_kg": len(rule),
            "llm_or_rule": len(llm_rule),
        },
        "rule_vs_temporal": compare_graphs(reference, rule),
        "llm_rule_vs_temporal": compare_graphs(reference, llm_rule),
        "pagerank": {
            "ordinary": ordinary_pagerank(graph),
            "temporal": graph.temporal_pagerank(),
        },
    }


def _llm_judge_edges(edges: List[CausalEdge], n: int = 5) -> float:
    from ..agents.llm_provider import LLMProvider

    provider = LLMProvider()
    scores = []
    for edge in edges[: min(n, len(edges))]:
        try:
            raw = provider.chat(
                system="你是金融因果边裁判。只输出1到5的整数。",
                user=(
                    f"判断这条因果边的合理性：{edge.source} -[{edge.relation}]-> {edge.target}，"
                    f"时间{edge.time}，证据{edge.evidence_ids}"
                ),
                temperature=0.0,
                use_cache=True,
            ).content.strip()
            score = float(raw[:1])
            if 1 <= score <= 5:
                scores.append(score)
        except Exception:
            continue
    return float(sum(scores) / max(len(scores), 1)) if scores else 0.0
