"""
Phase 0/1 logic self-test (no GPU, no Ollama, no faiss needed).
Verifies: firewall date filtering + source-index integrity + scalar parity.
Run:  python test_firewall_logic.py
"""
from firewall import WalkForwardContext, _to_date, _to_unix
from contracts import TransmissionMap


class Chunk:
    def __init__(self, cid, pub, meta=None):
        self.chunk_id = cid
        self.pub_time = _to_unix(_to_date(pub)) if pub else None
        self.metadata = meta or {}


class Idx:
    def __init__(self):
        self._chunks = []

    def add(self, cs):
        self._chunks.extend(cs)


class MockRet:
    def __init__(self, kg=None):
        self.vector_index = Idx()
        self.bm25_index = Idx()
        self.kg = kg

    def index_documents(self, cs):
        self.vector_index.add(cs)
        self.bm25_index.add(cs)

    def retrieve(self, q, entity_names=None, step_context=""):
        class P:
            pass
        p = P()
        p.chunks = list(self.vector_index._chunks)
        return p


def test_firewall():
    src = MockRet()
    src.index_documents([
        Chunk("past1", "2020-01-01"),
        Chunk("past2", "2024-07-21"),
        Chunk("eventday", "2024-07-22"),
        Chunk("future", "2024-07-25"),
        Chunk("undated", None),
        Chunk("timeless", "2025-01-01", {"valid_from": "2019-01-01"}),
    ])
    ctx = WalkForwardContext(event_date="2024-07-22", car_csv="NONE")
    fr = ctx.wrap_retriever(src)
    vis = {c.chunk_id for c in fr.visible}
    assert vis == {"past1", "past2", "timeless"}, vis
    fr.assert_no_future_leakage(fr.retrieve("q"))
    assert len(src.vector_index._chunks) == 6, "source index corrupted!"
    print("FIREWALL OK  visible =", sorted(vis))


def _v3_exog_factors(steps):
    """Reference impl copied from v3 mechanism_agent.to_exogenous_factors."""
    mag = {"高": 0.9, "中": 0.5, "低": 0.2, "未知": 0.0}
    factors = {}
    for s in steps:
        d = 1.0 if s.get("impact_direction") == "+" else -1.0
        m = mag.get(s.get("impact_magnitude", "未知"), 0.0)
        factors[s.get("entity", "")] = d * m * s.get("confidence", 0.5)
    return factors


def test_contract_parity():
    # Adversarial cases: duplicate entity (last-write-wins), unknown direction
    # (v3 maps to -1), empty direction (v3 -> -1), 未知 magnitude (0.0).
    class FakeChain:
        steps = [
            {"entity": "银行", "impact_direction": "+", "impact_magnitude": "高", "confidence": 0.8},
            {"entity": "地产", "impact_direction": "-", "impact_magnitude": "中", "confidence": 0.6},
            {"entity": "地产", "impact_direction": "+", "impact_magnitude": "低", "confidence": 0.9},  # duplicate -> overwrite
            {"entity": "券商", "impact_direction": "?", "impact_magnitude": "中", "confidence": 0.5},  # unknown -> -1
            {"entity": "钢铁", "impact_direction": "", "impact_magnitude": "未知", "confidence": 0.7},  # empty dir, 未知 mag -> 0
        ]
    tm = TransmissionMap.from_mechanism_chain(FakeChain(), industry_hint="")
    ref = _v3_exog_factors(FakeChain.steps)

    # 1) as_exog_factors byte-identical to v3 dict
    assert tm.as_exog_factors() == ref, (tm.as_exog_factors(), ref)
    # 2) net_signal == sum(v3 factors)
    assert abs(tm.project_to_scalar() - sum(ref.values())) < 1e-12, \
        (tm.project_to_scalar(), sum(ref.values()))
    # 3) event_strength == max-abs v3 factor
    v3_es = max(ref.values(), key=abs)
    assert abs(tm.event_strength() - v3_es) < 1e-12, (tm.event_strength(), v3_es)
    print("CONTRACT OK  net_signal =", round(tm.project_to_scalar(), 4),
          " event_strength =", round(tm.event_strength(), 4),
          " dispersion =", round(tm.dispersion(), 4),
          " (v3-exact parity on dup/unknown/empty cases)")


if __name__ == "__main__":
    test_firewall()
    test_contract_parity()
    print("ALL TESTS PASSED")
