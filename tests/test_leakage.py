import pandas as pd

from finmas.data.time_firewall import TimeFirewall
from finmas.rag.retrieval import HybridRetriever
from finmas.schemas import EvidenceChunk, EvidencePackage


def test_time_firewall_rows_are_strictly_before_event():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
            ),
            "value": [1, 2, 3, 4],
        }
    )
    visible = TimeFirewall("2024-01-03").rows_visible(frame)
    assert visible["date"].dt.date.max() < pd.Timestamp("2024-01-03").date()


def test_retriever_filters_future_chunks():
    chunks = [
        EvidenceChunk(
            chunk_id="old",
            text="old evidence",
            source="s",
            pub_time=1700000000,
        ),
        EvidenceChunk(
            chunk_id="new",
            text="future evidence",
            source="s",
            pub_time=1900000000,
        ),
    ]
    retriever = HybridRetriever()
    retriever.index(chunks)
    package = retriever.retrieve("evidence", top_k=4, as_of_date="2024-01-02")
    ids = [c.chunk_id for c in package.chunks]
    assert "new" not in ids

