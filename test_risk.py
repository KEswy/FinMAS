"""
Phase 4 risk-metrics test (no GPU/Ollama; pure numpy math).
Run:  python test_risk.py
"""
from evaluation.risk_metrics import (
    RiskHead, backtest_var, kupiec_pof, christoffersen_independence, pinball_loss,
)


def test_risk_head():
    rh = RiskHead(alpha=0.05, n_paths=2000, seed=42)
    # a mildly positive path with a symmetric interval
    p = [0.005, 0.004, 0.003, 0.002, 0.001]
    lo = [x - 0.02 for x in p]
    hi = [x + 0.02 for x in p]
    r0 = rh.compute(p, lo, hi, dispersion=0.0, k=5)
    r1 = rh.compute(p, lo, hi, dispersion=0.6, k=5)
    # MC VaR should not be the over-conservative sum of lower bounds; it should
    # still lie below the point forecast and inside the broad legacy interval.
    assert r0.event_VaR <= sum(p), r0.event_VaR
    assert r0.event_VaR >= sum(lo), (r0.event_VaR, sum(lo))
    assert 0.0 <= r0.expected_MDD <= 1.0
    assert r1.tail_prob > r0.tail_prob, "dispersion must widen tail"
    assert r0.risk_interval[0] < r0.risk_interval[1]
    assert r1.event_VaR < r0.event_VaR, "dispersion must widen VaR"
    # Legacy mode should exactly reproduce the previous sum-of-lower-bounds VaR.
    rh_legacy = RiskHead(alpha=0.05, method="legacy")
    r_legacy = rh_legacy.compute(p, lo, hi, dispersion=0.0, k=5)
    assert abs(r_legacy.event_VaR - sum(lo)) < 1e-9, r_legacy.event_VaR
    print(f"RISKHEAD OK  VaR={r0.event_VaR:.4f}  MDD={r0.expected_MDD:.4f}  "
          f"tail base={r0.tail_prob:.3f} disp={r1.tail_prob:.3f} "
          f"legacy={r_legacy.event_VaR:.4f}")


def test_kupiec():
    # well-calibrated: 5 breaches in 100 at alpha=0.05 -> should NOT reject
    lr_ok, p_ok = kupiec_pof(100, 5, 0.05)
    # badly calibrated: 20 breaches in 100 -> should reject (p small)
    lr_bad, p_bad = kupiec_pof(100, 20, 0.05)
    assert p_ok > 0.05, (lr_ok, p_ok)
    assert p_bad < 0.05, (lr_bad, p_bad)
    print(f"KUPIEC OK  calibrated p={p_ok:.3f} (keep)  over-breach p={p_bad:.4f} (reject)")


def test_christoffersen():
    # clustered breaches (all at end) vs spread out
    clustered = [0]*90 + [1]*10
    spread = ([0]*9 + [1]) * 10
    lr_c, p_c = christoffersen_independence(clustered)
    lr_s, p_s = christoffersen_independence(spread)
    # clustered should have larger LR (more evidence of dependence) than spread
    assert lr_c >= lr_s, (lr_c, lr_s)
    print(f"CHRISTOFFERSEN OK  clustered LR={lr_c:.3f} >= spread LR={lr_s:.3f}")


def test_pinball_and_backtest():
    # realized CARs, VaR set at a low quantile
    realized = [0.01, -0.005, 0.02, -0.03, 0.0, -0.06, 0.015, -0.01, 0.005, -0.02]
    var = [-0.05] * len(realized)   # VaR floor
    pin = pinball_loss(realized, var, 0.05)
    assert pin >= 0
    bt = backtest_var(realized, var, alpha=0.05)
    assert bt.n == 10
    assert bt.n_breach == sum(1 for r in realized if r < -0.05)  # =1 (-0.06)
    print(f"BACKTEST OK  n={bt.n} breaches={bt.n_breach} rate={bt.breach_rate:.2f} "
          f"kupiec_p={bt.kupiec_p:.3f} pinball={pin:.5f}")


if __name__ == "__main__":
    test_risk_head()
    test_kupiec()
    test_christoffersen()
    test_pinball_and_backtest()
    print("ALL RISK TESTS PASSED")
