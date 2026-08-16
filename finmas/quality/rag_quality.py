"""RAG ground-truth generation and quality evaluation."""

from __future__ import annotations

import json
import datetime as _dt
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..rag.retrieval import HybridRetriever, build_retriever, load_document_library
from ..schemas import EvidenceChunk
from .schemas import QualityMetric, QualityReport, Threshold


RAG_THRESHOLDS = [
    Threshold("recall_at_5", ">=", 0.70),
    Threshold("mrr", ">=", 0.55),
    Threshold("firewall_pass_rate", ">=", 1.0),
    Threshold("citation_coverage", ">=", 0.80),
]


def _to_unix(date_str: str) -> float:
    return time.mktime(time.strptime(str(date_str)[:10], "%Y-%m-%d"))


def build_rag_ground_truth(
    event_csv: str = "data/events/event_library.csv",
    doc_library: str = "data/events/doc_library_events.json",
    output: str = "data/eval/rag_ground_truth.json",
) -> List[Dict]:
    events = pd.read_csv(event_csv)
    events["event_date"] = events["event_date"].astype(str).str[:10]
    chunks = load_document_library(doc_library)
    by_event: Dict[str, List[EvidenceChunk]] = {}
    for chunk in chunks:
        event_date = str((chunk.metadata or {}).get("event_date", ""))[:10]
        if event_date:
            by_event.setdefault(event_date, []).append(chunk)

    samples = []
    for _, event in events.iterrows():
        event_date = str(event["event_date"])[:10]
        event_name = str(event.get("event_name", ""))
        description = str(event.get("description", ""))
        if not event_name:
            continue
        retrieval_as_of = (
            _dt.datetime.strptime(event_date, "%Y-%m-%d") + _dt.timedelta(days=1)
        ).strftime("%Y-%m-%d")
        cutoff = _to_unix(retrieval_as_of)
        relevant = [
            c.chunk_id
            for c in by_event.get(event_date, [])
            if c.pub_time <= cutoff
        ]
        if not relevant:
            continue
        samples.append(
            {
                "query": f"{event_name} {description}".strip(),
                "event_date": event_date,
                "retrieval_as_of_date": retrieval_as_of,
                "event_type": str(event.get("event_type", "")),
                "relevant_chunk_ids": sorted(set(relevant)),
            }
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return samples


def evaluate_rag(
    retriever: HybridRetriever,
    ground_truth_path: str = "data/eval/rag_ground_truth.json",
    top_k: int = 5,
) -> QualityReport:
    path = Path(ground_truth_path)
    if not path.exists():
        build_rag_ground_truth(output=str(path))
    samples = json.loads(path.read_text(encoding="utf-8"))
    recall_values = []
    rrs = []
    firewall_ok = True
    citation_values = []

    for sample in samples:
        relevant = set(sample["relevant_chunk_ids"])
        package = retriever.retrieve(
            sample["query"],
            top_k=top_k,
            as_of_date=sample.get("retrieval_as_of_date", sample["event_date"]),
        )
        retrieved = [c.chunk_id for c in package.chunks]
        hit = set(retrieved) & relevant
        recall_values.append(len(hit) / len(relevant))

        rr = 0.0
        for rank, cid in enumerate(retrieved, start=1):
            if cid in relevant:
                rr = 1.0 / rank
                break
        rrs.append(rr)

        firewall_ok = firewall_ok and package.passed_time_firewall
        citation_values.append(
            len(hit) / max(min(len(relevant), len(retrieved)), 1)
        )

    report = QualityReport(module="rag", n_samples=len(samples))
    observed = {
        "recall_at_5": float(sum(recall_values) / max(len(recall_values), 1)),
        "mrr": float(sum(rrs) / max(len(rrs), 1)),
        "firewall_pass_rate": float(firewall_ok),
        "citation_coverage": float(sum(citation_values) / max(len(citation_values), 1)),
    }
    for threshold in RAG_THRESHOLDS:
        value = observed.get(threshold.name, 0.0)
        report.metrics.append(
            QualityMetric(
                name=threshold.name,
                value=value,
                passed=threshold.check(value),
                threshold=threshold.value,
                operator=threshold.operator,
            )
        )
    return report.finalize()
