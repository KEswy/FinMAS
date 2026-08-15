"""
Batch VaR backtest across the ablation × seed matrix (v4 Phase-4 evidence).

For base/irf/all × seeds 42/123/2024, runs the pooled VaR backtest
(Kupiec POF / Christoffersen independence / pinball / breach rate) on each CSV
and reports a per-(config,seed) table plus a 3-seed aggregate. This is the
Phase-4 "risk is a first-class objective, independent of direction" evidence:
even at a ~60.5% direction ceiling, is the event-VaR well-calibrated?

Reuses evaluation.risk_metrics.backtest_var and the exact per-seed file
matching from analyze_multiseed (seed 42 = no _seed suffix).

Run:  python scripts/analyze_risk_matrix.py
"""
from __future__ import annotations
import os, glob, re, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)   # so `import evaluation` resolves
import numpy as np
import pandas as pd
from evaluation.risk_metrics import backtest_var

CONFIGS = ["base", "irf", "all"]
SEEDS = ["42", "123", "2024"]
ALPHA = 0.05


def find_csv(cfg: str, seed: str):
    if seed == "42":
        cands = [p for p in glob.glob(
            f"data/processed/all_experiment_results_qwen2.5_wf_{cfg}_*.csv")
            if not re.search(r"_seed\d+_", p)]
    else:
        cands = glob.glob(
            f"data/processed/all_experiment_results_qwen2.5_wf_{cfg}_seed{seed}_*.csv")
    return max(cands, key=os.path.getmtime) if cands else None


def run_one(path):
    d = pd.read_csv(path)
    if "event_VaR" not in d.columns:
        return None
    d = d.dropna(subset=["event_VaR", "real_CAR"])
    realized = d["real_CAR"].astype(float).tolist()
    var_pred = d["event_VaR"].astype(float).tolist()
    rmdd = pmdd = None
    if "expected_MDD" in d.columns:
        rmdd = [max(-x, 0.0) for x in realized]
        pmdd = d["expected_MDD"].astype(float).tolist()
    return backtest_var(realized, var_pred, alpha=ALPHA,
                        realized_mdd=rmdd, pred_mdd=pmdd)


def main() -> int:
    print("=" * 86)
    print(f"VaR backtest matrix (α={ALPHA}, independent of direction accuracy)")
    print("=" * 86)
    print(f"{'config':<6}{'seed':<7}{'breach%':>9}{'Kupiec p':>10}{'Christ p':>10}"
          f"{'pinball':>10}{'MDD MAE':>10}  {'coverage':>10}")
    print("-" * 86)
    agg = {c: {"breach": [], "kp": [], "pin": [], "mdd": []} for c in CONFIGS}
    for cfg in CONFIGS:
        for s in SEEDS:
            p = find_csv(cfg, s)
            if not p:
                print(f"{cfg:<6}{s:<7}{'MISSING':>9}")
                continue
            bt = run_one(p)
            if bt is None:
                print(f"{cfg:<6}{s:<7}{'no VaR col':>9}")
                continue
            cov = "PASS" if bt.passes_kupiec else "FAIL"
            print(f"{cfg:<6}{s:<7}{bt.breach_rate*100:>8.1f}%{bt.kupiec_p:>10.3f}"
                  f"{bt.christoffersen_p:>10.3f}{bt.pinball_loss:>10.5f}"
                  f"{bt.mdd_mae:>10.4f}  {cov:>10}")
            agg[cfg]["breach"].append(bt.breach_rate)
            agg[cfg]["kp"].append(bt.kupiec_p)
            agg[cfg]["pin"].append(bt.pinball_loss)
            agg[cfg]["mdd"].append(bt.mdd_mae)

    print("\n" + "=" * 86)
    print("3-seed aggregate")
    print("=" * 86)
    print(f"{'config':<6}{'breach% mean':>16}{'Kupiec p mean':>16}"
          f"{'pinball mean':>14}{'MDD MAE mean':>14}")
    for cfg in CONFIGS:
        a = agg[cfg]
        if not a["breach"]:
            continue
        print(f"{cfg:<6}{np.mean(a['breach'])*100:>15.1f}%{np.mean(a['kp']):>16.3f}"
              f"{np.mean(a['pin']):>14.5f}{np.mean(a['mdd']):>14.4f}")
    print(f"\n判读：breach% 应接近 α={ALPHA*100:.0f}%（校准良好）；Kupiec p>0.05=覆盖率通过；")
    print("Christ p>0.05=违约不聚簇。n=168 小样本尾部，Kupiec FAIL 是诚实发现（驱动扩样），非 bug。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
