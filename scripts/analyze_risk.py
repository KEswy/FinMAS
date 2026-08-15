"""
Phase 4 risk evaluation — pooled VaR backtest across all events (independent
of direction accuracy). Reads a walk-forward result CSV (must contain the
Phase-4 columns event_VaR / real_CAR / expected_MDD) and reports:
  - breach rate vs nominal alpha
  - Kupiec POF (unconditional coverage) test
  - Christoffersen independence test
  - pinball (quantile) loss
  - drawdown-calibration proxy

Run:  python scripts/analyze_risk.py <results.csv>
      python scripts/analyze_risk.py            # auto-picks latest *_wf_*.csv
"""
from __future__ import annotations
import os, sys, glob
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from evaluation.risk_metrics import backtest_var

ALPHA = 0.05


def _pick_csv() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    cands = glob.glob("data/processed/all_experiment_results_*_wf_*.csv")
    if not cands:
        print("no walk-forward CSV found; pass a path explicitly", file=sys.stderr)
        sys.exit(1)
    return max(cands, key=os.path.getmtime)


def main() -> int:
    path = _pick_csv()
    df = pd.read_csv(path)
    print(f"[risk] {path}  ({len(df)} events)")
    if "event_VaR" not in df.columns:
        print("ERROR: CSV lacks Phase-4 risk columns (event_VaR). "
              "Re-run run_walk_forward.py after the Phase-4 patch.", file=sys.stderr)
        return 1

    d = df.dropna(subset=["event_VaR", "real_CAR"]).copy()
    realized = d["real_CAR"].astype(float).tolist()
    var_pred = d["event_VaR"].astype(float).tolist()

    rmdd = pmdd = None
    if "expected_MDD" in d.columns:
        # realized MDD proxy: |negative part of real_CAR| (single-terminal proxy)
        rmdd = [max(-x, 0.0) for x in realized]
        pmdd = d["expected_MDD"].astype(float).tolist()

    bt = backtest_var(realized, var_pred, alpha=ALPHA,
                      realized_mdd=rmdd, pred_mdd=pmdd)

    print("=" * 56)
    print("Phase 4 — VaR backtest (independent of direction)")
    print("=" * 56)
    print(f"  events           : {bt.n}")
    print(f"  VaR breaches     : {bt.n_breach}  (real_CAR < predicted VaR)")
    print(f"  breach rate      : {bt.breach_rate:.3f}   nominal α = {bt.expected_rate}")
    print(f"  Kupiec POF       : LR={bt.kupiec_LR:.3f}  p={bt.kupiec_p:.3f}  "
          f"{'PASS (coverage ok)' if bt.passes_kupiec else 'FAIL (miscalibrated)'}")
    print(f"  Christoffersen   : LR={bt.christoffersen_LR:.3f}  p={bt.christoffersen_p:.3f}  "
          f"{'(breaches independent)' if bt.christoffersen_p > 0.05 else '(breaches clustered)'}")
    print(f"  pinball loss     : {bt.pinball_loss:.5f}")
    if bt.mdd_mae == bt.mdd_mae:  # not nan
        print(f"  MDD calib MAE    : {bt.mdd_mae:.4f}  (|pred E[MDD] − realized downside|)")
    print("=" * 56)
    print("Note: small-sample tail — report CIs; a Kupiec FAIL at n=168 is itself")
    print("an honest finding motivating data expansion, not a bug.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
