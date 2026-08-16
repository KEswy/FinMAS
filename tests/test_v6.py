import json
from pathlib import Path

from finmas.graph import FinMASGraph
from finmas.pipeline import FinMASPipeline
from finmas.quality import build_rag_ground_truth
from finmas.quality.reports import generate_all_quality_reports
from finmas.schemas import EventInput


def test_build_rag_ground_truth_has_samples(tmp_path):
    output = tmp_path / "rag_ground_truth.json"
    samples = build_rag_ground_truth(output=str(output))
    assert len(samples) > 0
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed
    assert "relevant_chunk_ids" in parsed[0]


def test_graph_run_no_llm_produces_decision():
    event = EventInput(
        event_date="2024-07-22",
        event_text="LPR下调",
        event_type="monetary_policy",
        industry_code="801780",
    )
    graph = FinMASGraph(
        pipeline=FinMASPipeline(use_llm=False, horizon=5),
        debate_rounds=1,
    )
    state = graph.run(event)
    assert state.decision is not None
    assert state.decision.prob_up >= 0.0
    assert state.trace


def test_generate_all_quality_reports(tmp_path):
    reports = generate_all_quality_reports(
        result_csv="data/processed/all_experiment_results_finmas_v5_final_ensemble.csv",
        risk_json="data/processed/finmas_v5_final_risk_backtest.json",
        output_dir=str(tmp_path),
    )
    assert "llm" in reports
    assert "fusion" in reports
    assert "risk" in reports
    assert (tmp_path / "quality_fusion.json").exists()

