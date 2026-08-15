import numpy as np

from finmas.data.time_firewall import TimeFirewall
from finmas.fusion.decision import DirectionFusion
from finmas.models.pool import MomentumModel, ModelPool
from finmas.risk.engine import backtest_var, evt_var_es, historical_var_es
from finmas.schemas import (
    Direction,
    DirectionDecision,
    EventInput,
    FactorSignal,
    SignalBundle,
)
from finmas.agents.text_factor_extractor import TextFactorExtractor


def test_direction_decision_roundtrip():
    decision = DirectionDecision(
        event_date="2024-01-02",
        industry_code="801780",
        horizon=5,
        prob_up=0.71,
        final_direction=Direction.UP.value,
        confidence=0.42,
    )
    data = decision.to_dict()
    assert data["prob_up"] == 0.71
    assert data["final_direction"] == "+"


def test_time_firewall_keeps_only_past_chunks():
    from finmas.schemas import EvidenceChunk, EvidencePackage

    firewall = TimeFirewall("2024-07-22")
    old = EvidenceChunk("old", "old", "s", pub_time=1721600000)
    new = EvidenceChunk("new", "new", "s", pub_time=1722000000)
    package = EvidencePackage("q", chunks=[old, new])
    filtered = firewall.filter_evidence(package)
    assert [c.chunk_id for c in filtered.chunks] == ["old"]


def test_historical_and_evt_risk():
    returns = np.linspace(-0.05, 0.05, 100)
    var, es = historical_var_es(returns, 0.05)
    assert var < 0
    evt_var, evt_es = evt_var_es(returns, 0.05)
    assert np.isfinite(evt_var)
    stats = backtest_var(returns, np.full_like(returns, var))
    assert "breach_rate" in stats


def test_momentum_model():
    x = np.random.default_rng(0).normal(0.001, 0.01, size=(80, 6))
    y = np.random.default_rng(1).normal(0, 0.01, size=(80, 5))
    model = MomentumModel(horizon=5).fit(x, y)
    pred = model.predict(x[-20:])
    assert pred.mean.shape == (5,)
    assert 0.0 <= pred.prob_up <= 1.0


def test_model_pool_fits_and_predicts():
    x = np.random.default_rng(2).normal(0.001, 0.01, size=(160, 6))
    y = x[:, :1]
    pool = ModelPool(horizon=5, input_len=60, output_len=5).fit(x, y)
    pred = pool.predict(x[-60:])
    assert pred.mean.shape == (5,)


def test_direction_fusion_abstains_on_high_disagreement():
    event = EventInput("2024-01-02", "降息", "monetary_policy", "801780")
    bundle = SignalBundle(
        event=event,
        llm_factors=[FactorSignal("a", 0.5, 0.8, "+")],
        time_series={"prob_up": 0.8, "signal": 0.01},
        disagreement=0.9,
    )
    decision = DirectionFusion().decide(bundle)
    assert decision.abstain is True


def test_text_factor_extractor_emits_signed_factors():
    event = EventInput("2024-01-02", "央行降息降准", "monetary_policy", "801780")
    factors = TextFactorExtractor().extract(event)
    assert factors
    assert any(f.direction == "+" for f in factors)
