from finmas.causal.extractor import CausalExtractor
from finmas.causal.graph import TemporalCausalGraph
from finmas.causal.schemas import CausalEdge
from finmas.causal.quality import compute_quality


def test_extractor_fallback_edges():
    edges = CausalExtractor(use_llm=False).extract(
        "央行降息并降准",
        "2024-01-02",
        ["801780", "801180"],
    )
    assert edges
    assert any(e.relation == "affects" for e in edges)


def test_temporal_graph_pagerank_and_paths(tmp_path):
    graph = TemporalCausalGraph(str(tmp_path / "graph.sqlite"))
    graph.add_edge(CausalEdge("event", "bank", "affects", "2024-01-01", 1.0, 0.9, ["e1"]))
    graph.add_edge(CausalEdge("bank", "market", "affects", "2024-01-02", 1.0, 0.8, ["e2"]))
    ranks = graph.temporal_pagerank(as_of="2024-01-03")
    assert "market" in ranks
    paths = graph.shortest_causal_paths("event", "market", as_of="2024-01-03")
    assert paths
    assert paths[0].nodes == ["event", "bank", "market"]


def test_causal_quality(tmp_path):
    graph = TemporalCausalGraph(str(tmp_path / "graph.sqlite"))
    graph.add_edge(CausalEdge("event", "industry", "affects", "2024-01-01", 1.0, 0.7, ["e"]))
    report = compute_quality(graph)
    assert report.edge_count == 1
    assert report.evidence_coverage == 1.0

