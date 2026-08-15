"""
Phase 2 fundamentals-skill test: sensible output + TEMPORAL ISOLATION.
Needs pandas + data/raw/industry_daily.csv (no GPU/Ollama).
Run:  python test_fundamentals.py
"""
import pandas as pd
from agents.fundamentals_agent import FundamentalsSkill
from windowing import build_feature_frame


def test_probe_sane():
    sk = FundamentalsSkill(lookback=20)
    pr = sk.probe("801780", "2024-07-22")   # banks, lots of prior history
    assert pr.n_hist > 20, pr.n_hist
    assert -1.0 <= pr.pre_state <= 1.0, pr.pre_state
    assert 0.0 <= pr.confidence <= 1.0, pr.confidence
    assert len(pr.layers) == 3, pr.layers
    print(f"PROBE OK  n_hist={pr.n_hist}  pre_state={pr.pre_state:+.3f} "
          f"conf={pr.confidence:.3f}  mom20={pr.momentum_20d:+.3f}")


def test_temporal_isolation():
    """The probe at t must use ONLY rows strictly before t. We verify by
    truncating the feature frame at t and confirming the probe is identical —
    i.e. future rows cannot influence it."""
    sk_full = FundamentalsSkill(lookback=20)
    pr_full = sk_full.probe("801780", "2024-07-22")

    # build a skill whose frame is pre-truncated to < t, then probe again
    fdf = build_feature_frame("801780")
    t = pd.to_datetime("2024-07-22")
    truncated = fdf[fdf.index < t].copy()
    sk_trunc = FundamentalsSkill(lookback=20)
    sk_trunc._frames["801780"] = truncated      # inject <t-only frame
    pr_trunc = sk_trunc.probe("801780", "2024-07-22")

    for f in ("momentum_20d", "rel_strength", "vol_regime",
              "valuation_pos", "volume_trend", "pre_state"):
        a, b = getattr(pr_full, f), getattr(pr_trunc, f)
        assert abs(a - b) < 1e-9, (f, a, b)
    print(f"ISOLATION OK  full==truncated on all signals "
          f"(future rows do not leak into probe at t)")


def test_cold_start():
    sk = FundamentalsSkill(lookback=20)
    pr = sk.probe("801780", "2019-01-10")   # near data start -> insufficient <t
    assert pr.pre_state == 0.0 and pr.confidence == 0.0, (pr.n_hist, pr.pre_state)
    print(f"COLD-START OK  early event n_hist={pr.n_hist} -> neutral probe")


def test_valuation_sane():
    sk = FundamentalsSkill()
    vp = sk.probe_valuation("801780", "2024-07-22")
    if vp.n_hist == 0:
        print("VALUATION SKIP  (industry_fundamentals.csv 未抓；先跑 fetch_industry_fundamentals.py)")
        return False
    assert vp.n_hist > 60, vp.n_hist
    assert 0.0 <= vp.pe_percentile <= 1.0, vp.pe_percentile
    assert -1.0 <= vp.valuation_signal <= 1.0, vp.valuation_signal
    assert 0.0 <= vp.confidence <= 1.0, vp.confidence
    print(f"VALUATION OK  {vp.note}  signal={vp.valuation_signal:+.3f} conf={vp.confidence:.3f}")
    return True


def test_valuation_isolation():
    """probe_valuation at t must use ONLY rows < t: truncating the table at t
    must not change the result (future PE rows cannot leak into the percentile)."""
    sk_full = FundamentalsSkill()
    if sk_full._val_table() is None:
        print("VALUATION-ISO SKIP  (no valuation csv)")
        return
    vp_full = sk_full.probe_valuation("801780", "2024-07-22")
    if vp_full.n_hist == 0:
        print("VALUATION-ISO SKIP")
        return
    t = pd.to_datetime("2024-07-22")
    trunc = sk_full._val_table()
    trunc = trunc[trunc["_d"] < t].copy()
    sk_tr = FundamentalsSkill()
    sk_tr._val_df = trunc
    vp_tr = sk_tr.probe_valuation("801780", "2024-07-22")
    for f in ("pe_percentile", "pb_percentile", "dividend_yield",
              "pe_momentum_20d", "valuation_signal"):
        a, b = getattr(vp_full, f), getattr(vp_tr, f)
        assert abs(a - b) < 1e-9, (f, a, b)
    print("VALUATION-ISO OK  full==truncated (future PE rows do not leak)")


if __name__ == "__main__":
    test_probe_sane()
    test_temporal_isolation()
    test_cold_start()
    test_valuation_sane()
    test_valuation_isolation()
    print("ALL FUNDAMENTALS TESTS PASSED")
