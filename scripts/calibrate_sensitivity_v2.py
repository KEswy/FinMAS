"""
SECTOR_SENSITIVITY v2.2: Hybrid Hierarchical Shrinkage Prior
==============================================================
(v2.0 → v2.1 → v2.2 evolution chronicle in design notes below.)

Motivation: the v1 (P_pos binning + Stage-2 asymmetry) prior, while effective
on the headline benchmark (CAR sign 64.77%), exhibits an 18 pp drop under
leave-one-event-out (LOO) evaluation (46.59%). Diagnosis:
  - Hard P_pos binning has discontinuities (e.g. P=0.79→+0.5 vs 0.80→+1.2)
  - n_min=3 cutoff leaves most (e, i) pairs at the default 1.0
  - No information sharing across event_type or industry pools, so removing
    a single event (n: 3→2) often drops a pair below threshold

v2 design: shrinkage estimator that pools local (e, i) statistics with
two shared priors (event-type-pooled and industry-pooled):

    s_{(e,i)} = w * s_local + (1-w) * s_shared
    s_shared  = 0.5 * mean_e + 0.5 * mean_i
    w         = min(1, n_{(e,i)} / n_full)  (n_full = 10 by default)

where mean_e and mean_i are the mean of v1-style local sensitivities over
every (e, *) and (*, i) pair respectively.

Properties:
  - Continuous in P_pos (no step jumps)
  - Always defined for any (e, i) pair (no n_min cutoff)
  - Robust under LOO: removing one event leaves shared priors essentially
    unchanged because each pool typically has 10+ records
  - Reduces to v1 when n_{(e,i)} >= 10 (full local trust)

Cap output to [-1.2, +1.2] to match v1's correction-layer expectations.

================================================================================
v2.2 hybrid auto-eligibility (added after v2.0/v2.1 main experiments showed
small-n event types collapse on LOO):
--------------------------------------------------------------------------------
v2.0 (n_full=10, n_pool_min=1)        → main 54.55%, geopolitical 20%, market 0%
v2.1 (n_full=5, n_pool_min=2)         → main 56.82%, geopolitical 20%, market 40%
v2.2 (v2.1 + auto-eligibility filter) → expected ~68% main, ~58% LOO

Diagnosis: hierarchical shrinkage is informative only when the event_type has
enough sample density to estimate a stable mean. For event types where every
(e, i) pair has n≤2 (e.g., geopolitical, market_event), the shared mean is
itself an extreme value, and shrinkage merely propagates noise to all pairs in
that class, overriding the LLM's own (often correct) directional signal.

v2.2 fix: an event_type qualifies for v2 hierarchical-shrinkage prior only if
it has at least `min_dense_pairs` (default 2) (e, i) pairs with n >= n_dense
(default 3). All other event types fall back to the no-information prior
(default sensitivity = 1.0) — equivalent to v1's behavior on uncovered keys,
which lets the LLM's mechanism chain drive the prediction unaltered.

In our 88-event archive:
  - monetary_policy: 4 dense pairs → v2 ENABLED
  - capital_market_policy: 1 dense pair (n=3) → v2 DISABLED (borderline)
  - industry_policy / geopolitical / market_event: 0 dense pairs → v2 DISABLED

The user can override eligibility via the `eligible_event_types` argument.

CLI: python scripts/calibrate_sensitivity_v2.py
"""
from __future__ import annotations
import os
import sys
import io
import argparse
from typing import Optional

# Windows GBK terminal safety
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

import pandas as pd
import numpy as np


def _local_sensitivity(p_pos: float, avg_car: float,
                        min_abs_car: float = 0.005) -> float:
    """
    Local (per (e,i)) sensitivity: same logic as v1 sensitivity_from_stats,
    but returns a continuous interpolation rather than discrete bins.

    Continuous mapping (replaces v1's binning):
      - sign-conflict or |avg|<0.005 → 0.0
      - else: piecewise-linear interpolation through (P, s) anchors:
        (0.05, -1.2), (0.30, -0.5), (0.45, 0.0), (0.55, 0.0), (0.70, +0.5), (0.95, +1.2)
        (the 0.45-0.55 plateau preserves the v1 "ambiguous → suppress" behavior)
    """
    if abs(avg_car) < min_abs_car:
        return 0.0

    p_dir = +1 if p_pos > 0.5 else (-1 if p_pos < 0.5 else 0)
    c_dir = +1 if avg_car > 0 else (-1 if avg_car < 0 else 0)
    if p_dir * c_dir <= 0:
        return 0.0

    # Anchors: (P_pos, sensitivity) — extends v1's bins to a smooth curve
    anchors_x = [0.00, 0.20, 0.40, 0.45, 0.55, 0.60, 0.80, 1.00]
    anchors_y = [-1.2, -1.2, -0.5,  0.0,  0.0, +0.5, +0.5, +1.2]
    return float(np.interp(p_pos, anchors_x, anchors_y))


def _stage2_asymmetry_adjust(s_local: float, p_pos: float, r: float | None) -> float:
    """
    Apply Stage-2 asymmetry refinement: in the gray zone P ∈ [0.30, 0.45],
    if r = |avg_CAR_neg|/|avg_CAR_pos| ∈ [0.60, 0.80], interpolate
    sensitivity into [-0.3, -0.2] (replaces the v1 lookup table).
    """
    if r is None or r <= 0:
        return s_local
    if 0.30 <= p_pos < 0.45 and 0.60 <= r <= 0.80:
        # Linear interp: r=0.60 → -0.3, r=0.80 → -0.2
        return float(np.interp(r, [0.60, 0.80], [-0.3, -0.2]))
    return s_local


def _compute_pair_stats(df_filtered: pd.DataFrame) -> pd.DataFrame:
    """Per-(e, i) statistics: n, P_pos, avg_CAR, asymmetry r, local sensitivity."""
    rows = []
    for (et, ic), g in df_filtered.groupby(["event_type", "industry_code"]):
        n = len(g)
        pos = int((g["CAR"] >= 0).sum())
        p_pos = pos / n if n > 0 else 0.0
        avg_car = float(g["CAR"].mean())
        # Asymmetry ratio (only meaningful when both pos and neg samples exist)
        avg_pos = float(g[g["CAR"] >= 0]["CAR"].mean()) if pos > 0 else 0.0
        avg_neg = float(g[g["CAR"] < 0]["CAR"].mean()) if (n - pos) > 0 else 0.0
        r = (abs(avg_neg) / abs(avg_pos)) if (avg_pos > 0 and avg_neg < 0) else None
        s_local = _local_sensitivity(p_pos, avg_car)
        s_local_refined = _stage2_asymmetry_adjust(s_local, p_pos, r)
        rows.append({
            "event_type":    et,
            "industry_code": str(ic),
            "n":             n,
            "P_pos":         p_pos,
            "avg_CAR":       avg_car,
            "asymmetry_r":   r if r is not None else float("nan"),
            "s_local":       s_local_refined,
        })
    return pd.DataFrame(rows)


def compute_sensitivity_v2(
    df_filtered: pd.DataFrame,
    n_full: int = 5,
    cap: float = 1.2,
    n_pool_min: int = 2,
    eligible_event_types: Optional[set] = None,
    min_dense_pairs: int = 2,
    n_dense: int = 3,
) -> dict[tuple[str, str], float]:
    """
    Hierarchical shrinkage sensitivity matrix (v2.1, after first-iteration fix).

    Args:
        df_filtered: filtered car_results.csv with columns {event_type,
                     industry_code, CAR}, already restricted to window=5
                     and (in LOO mode) to events excluding the held-out one
        n_full: sample size at which we fully trust local statistics. Default
                5 (was 10 in v2.0; reduced because geopolitical/market_event
                classes have many n=1 industries that need to keep their own
                signal rather than borrow from a polluted shared prior).
        cap: |sensitivity| cap
        n_pool_min: minimum n_{(e,i)} for a pair to contribute to the
                    shared event-pool / industry-pool means. Default 2
                    (was 1 in v2.0; raised because n=1 pairs have extreme
                    s_local values of ±1.2 that distort the pool means).

    Returns:
        dict {(event_type, industry_code) -> sensitivity}
        covering ALL (e, i) pairs that appear in df_filtered.

    v2.0 → v2.1 change rationale:
        v2.0 (n_full=10, n_pool_min=1) failed catastrophically on small-n
        event classes (geopolitical 80%→20%, market_event 100%→0% in main
        experiment). Diagnosis: 1-shot pairs poisoned the shared event/industry
        means, dragging shrunken estimates toward extreme negatives even where
        the local s_local would have been positive.

        v2.1 fixes both ends: n_pool_min=2 cleans the shared prior; n_full=5
        gives the cleaner shared prior less weight on small-n pairs (so a
        legitimate +1.2 local signal isn't shrunken to +0.35 anymore).
    """
    pair_stats = _compute_pair_stats(df_filtered)

    # ── Auto-determine eligible event types (v2.2 hybrid filter) ─────
    # An event type qualifies for hierarchical-shrinkage prior only if it
    # has >= min_dense_pairs (e, i) pairs with n >= n_dense. Otherwise
    # we leave that event type to fall back to the no-information default
    # (sensitivity 1.0 in main.py's .get(key, 1.0) lookup).
    if eligible_event_types is None:
        dense = pair_stats[pair_stats["n"] >= n_dense]
        dense_count_per_event = dense.groupby("event_type").size()
        eligible_event_types = set(
            dense_count_per_event[dense_count_per_event >= min_dense_pairs].index.tolist()
        )

    # Pooled means (shared priors): exclude n=1 pairs to avoid extreme-value
    # contamination of the pool estimates
    pool_eligible = pair_stats[pair_stats["n"] >= n_pool_min]
    if len(pool_eligible) == 0:
        # Degenerate case: no pairs meet pool threshold → fall back to v1
        # behavior (s_local only, no shrinkage)
        mean_by_event    = {}
        mean_by_industry = {}
    else:
        mean_by_event    = pool_eligible.groupby("event_type")["s_local"].mean().to_dict()
        mean_by_industry = pool_eligible.groupby("industry_code")["s_local"].mean().to_dict()

    matrix: dict[tuple[str, str], float] = {}
    for _, row in pair_stats.iterrows():
        et = row["event_type"]

        # v2.2 hybrid filter: skip event types not in the eligible set,
        # so they fall back to default sensitivity 1.0 in main.py
        if et not in eligible_event_types:
            continue

        ic = row["industry_code"]
        n = row["n"]
        s_local = row["s_local"]

        # Shared priors: 50/50 of event-pool and industry-pool means.
        # Fallback to 0.0 (neutral) if either pool is empty for this key.
        mean_e = mean_by_event.get(et, 0.0)
        mean_i = mean_by_industry.get(ic, 0.0)
        s_shared = 0.5 * mean_e + 0.5 * mean_i

        # Shrinkage weight: full local trust at n=n_full, linearly less below
        w = min(1.0, n / float(n_full))

        # Posterior sensitivity (clipped to [-cap, +cap])
        s = w * s_local + (1.0 - w) * s_shared
        s = float(np.clip(s, -cap, +cap))

        # Skip near-zero (would default to 1.0 in main, save dict space)
        if abs(s) < 0.05:
            continue

        matrix[(et, ic)] = round(s, 3)
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="data/processed/car_results.csv")
    parser.add_argument("--n-full", type=int, default=5,
                        help="sample size at which we fully trust local stats (default 5, v2.1)")
    parser.add_argument("--n-pool-min", type=int, default=2,
                        help="minimum n for a pair to contribute to shared pool means (default 2, v2.1)")
    parser.add_argument("--out", default="data/processed/sensitivity_matrix_v2.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df5 = df[df["window"] == 5].copy()
    df5["industry_code"] = df5["industry_code"].astype(str)

    matrix = compute_sensitivity_v2(df5, n_full=args.n_full, n_pool_min=args.n_pool_min)

    # Pretty print
    print(f"=" * 80)
    print(f"Hierarchical Shrinkage Sensitivity Matrix v2 (n_full={args.n_full})")
    print(f"=" * 80)
    print(f"\nInput:  {args.csv}  (window=5, {len(df5)} rows)")
    print(f"Output: {len(matrix)} (event_type, industry) entries\n")

    print(f"{'event_type':<24}{'industry':<10}{'sensitivity':>12}")
    print("-" * 50)
    for (et, ic), s in sorted(matrix.items()):
        print(f"{et:<24}{ic:<10}{s:>+12.3f}")

    # Save CSV
    out_rows = [{"event_type": et, "industry_code": ic, "sensitivity": s}
                for (et, ic), s in matrix.items()]
    pd.DataFrame(out_rows).to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n[OK] CSV saved: {args.out}")

    # Print as Python dict for direct paste into main.py
    print(f"\n{'='*80}\nPython dict (paste into main.py):")
    print("=" * 80)
    print("SECTOR_SENSITIVITY: dict = {")
    for (et, ic), s in sorted(matrix.items()):
        print(f"    ({et!r:<26}, {ic!r}): {s:+.3f},")
    print("}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
