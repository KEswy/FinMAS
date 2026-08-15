"""
Phase 3 transmission-skill blend logic test (no GPU/Ollama/pandas-CAR needed;
uses walk_forward=False so strength is not <t-recalibrated).
Run:  python test_transmission.py
"""
from agents.transmission_agent import TransmissionSkill, BLEND_LAMBDA


def main():
    sk = TransmissionSkill()  # loads data/irf_priors.json
    assert sk._priors, "irf_priors.json not loaded / empty"

    # 1) No IRF cell -> net_signal unchanged
    out, info = sk.blend_net_signal(2.0, "monetary_policy", "999999",
                                    "2024-07-22", walk_forward=False)
    assert out == 2.0 and info["irf"] == "none", (out, info)
    print("NO-CELL OK  (unknown industry passes through)")

    # 2) monetary×bank IRF=(+1, ~0.1): weak positive prior should NOT flip a
    #    strong positive LLM signal, only nudge it.
    d, s = sk.irf_signed("monetary_policy", "801780")
    out, info = sk.blend_net_signal(3.15, "monetary_policy", "801780",
                                    "2024-07-22", walk_forward=False)
    assert d == 1, ("bank dir", d)
    assert out > 0, ("weak+ prior must keep strong+ LLM positive", out, info)
    print(f"WEAK-PRIOR OK  bank dir={d} s={s}  net 3.15 -> {info['net_out']} (stays +)")

    # 3) market_event×non-bank IRF=(-1, strong): if LLM wrongly says +,
    #    a strong negative prior should pull the blend down (toward correct -).
    d2, s2 = sk.irf_signed("market_event", "801790")
    out3, info3 = sk.blend_net_signal(1.5, "market_event", "801790",
                                     "2025-04-07", walk_forward=False)
    assert d2 == -1, ("non-bank market_event dir", d2)
    # blend must be strictly below the pure-LLM mapping (pulled toward negative)
    out_noirf, _ = sk.blend_net_signal(1.5, "market_event", "999999",
                                       "2025-04-07", walk_forward=False)
    assert out3 < out_noirf, ("strong- prior must pull signal down", out3, out_noirf)
    print(f"CONFLICT-PULL OK  mkt non-bank dir={d2} s={s2}  "
          f"LLM+1.5 -> {info3['net_out']} (< no-IRF {round(out_noirf,3)})")

    # 4) lambda is the pre-registered fixed weight, not fitted
    assert abs(sk.blend_lambda - BLEND_LAMBDA) < 1e-9
    print(f"LAMBDA OK  fixed blend λ={sk.blend_lambda}")

    print("ALL TRANSMISSION TESTS PASSED")


if __name__ == "__main__":
    main()
