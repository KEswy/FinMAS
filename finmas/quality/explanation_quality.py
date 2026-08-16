"""Explanation and knowledge-discovery quality metrics for v7."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Optional

import networkx as nx

from ..agents.llm_provider import LLMProvider
from ..sim.schemas import SimulationState
from .schemas import QualityMetric, QualityReport


def _state_from_file(path: str) -> SimulationState:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return SimulationState.from_dict(raw)


def compute_explanation_metrics(
    state: SimulationState,
    llm_judge: bool = False,
    judge_n: int = 5,
) -> QualityReport:
    report = QualityReport(module="explanation", n_samples=len(state.timeline))
    opinions = [o for e in state.timeline for o in e.opinions]
    paths = [p for e in state.timeline for p in e.causal_paths]

    evidence_coverage = (
        sum(1 for o in opinions if o.evidence_ids) / max(len(opinions), 1)
    )
    path_evidence_coverage = (
        sum(1 for p in paths if p.evidence_ids) / max(len(paths), 1)
    )
    industry_mention_rate = (
        sum(
            1
            for e in state.timeline
            if any(code in e.narrative for code in e.tick.industry_returns)
        )
        / max(len(state.timeline), 1)
    )
    event_days = [e for e in state.timeline if e.tick.triggered_event]
    event_mention_rate = (
        sum(
            1
            for e in event_days
            if e.tick.triggered_event[:20] in e.narrative
        )
        / max(len(event_days), 1)
    )
    avg_confidence = statistics.mean(o.confidence for o in opinions) if opinions else 0.0

    graph = nx.DiGraph()
    for p in paths:
        graph.add_edge(p.source, p.target)
    cycle_rate = 1.0 if list(nx.simple_cycles(graph)) else 0.0

    metrics = [
        QualityMetric("evidence_coverage", evidence_coverage,
                      evidence_coverage >= 0.80, 0.80, ">="),
        QualityMetric("path_evidence_coverage", path_evidence_coverage,
                      path_evidence_coverage >= 0.95, 0.95, ">="),
        QualityMetric("industry_mention_rate", industry_mention_rate,
                      industry_mention_rate >= 0.90, 0.90, ">="),
        QualityMetric("event_mention_rate", event_mention_rate,
                      event_mention_rate >= 0.50, 0.50, ">="),
        QualityMetric("avg_confidence", avg_confidence,
                      avg_confidence >= 0.40, 0.40, ">="),
        QualityMetric("cycle_rate", cycle_rate,
                      cycle_rate <= 0.10, 0.10, "<="),
    ]
    report.metrics.extend(metrics)

    if llm_judge and state.timeline:
        report.metrics.append(
            QualityMetric(
                "llm_judge_avg_score",
                _llm_judge_score(state, judge_n),
                True,
                3.0,
                ">=",
            )
        )
    return report.finalize()


def _llm_judge_score(state: SimulationState, n: int) -> float:
    provider = LLMProvider()
    samples = state.timeline[: min(n, len(state.timeline))]
    scores = []
    for entry in samples:
        prompt = (
            "请给下面这段金融叙事打分(1-5)：事实性、因果清晰度、是否夸大。"
            "只输出一个数字。\n"
            f"日期：{entry.date}\n市场收益：{entry.tick.market_return:+.4f}\n"
            f"叙事：{entry.narrative}"
        )
        try:
            raw = provider.chat(
                system="你是金融解释质量裁判。只输出1到5的整数。",
                user=prompt,
                temperature=0.0,
                use_cache=True,
            ).content.strip()
            score = float(raw[:1])
            if 1 <= score <= 5:
                scores.append(score)
        except Exception:
            continue
    return float(statistics.mean(scores)) if scores else 0.0
