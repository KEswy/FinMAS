"""Distribution-free and EVT risk estimation with horizon calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..schemas import RiskReport, RiskStats


def _as_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def historical_var_es(returns: Sequence[float], alpha: float) -> tuple[float, float]:
    arr = _as_array(returns)
    if len(arr) == 0:
        return float("nan"), float("nan")
    var = float(np.quantile(arr, alpha))
    tail = arr[arr <= var]
    es = float(np.mean(tail)) if len(tail) else var
    return var, es


def evt_var_es(returns: Sequence[float], alpha: float, tail_frac: float = 0.10) -> tuple[float, float]:
    arr = _as_array(returns)
    if len(arr) < 30:
        return historical_var_es(arr, alpha)
    losses = -arr
    threshold = float(np.quantile(losses, 1.0 - tail_frac))
    excess = losses[losses > threshold] - threshold
    if len(excess) < 10:
        return historical_var_es(arr, alpha)
    mean_excess = float(np.mean(excess))
    var_excess = float(np.var(excess, ddof=1))
    if var_excess <= 1e-12 or mean_excess <= 0:
        return historical_var_es(arr, alpha)
    shape = 0.5 * ((mean_excess**2 / var_excess) - 1.0)
    scale = 0.5 * mean_excess * ((mean_excess**2 / var_excess) + 1.0)
    n = len(losses)
    n_u = len(excess)
    if abs(shape) < 1e-6:
        loss_var = threshold - scale * np.log((n / n_u) * alpha)
        loss_es = loss_var + scale
    else:
        tail_prob = (n / n_u) * alpha
        loss_var = threshold + (scale / shape) * ((tail_prob ** (-shape)) - 1.0)
        loss_es = (loss_var + scale - shape * threshold) / (1.0 - shape)
    return float(-loss_var), float(-loss_es)


def max_drawdown(path: Sequence[float]) -> float:
    nav = 1.0 + np.asarray(path, dtype=float)
    roll_max = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
    dd = (roll_max - nav) / np.maximum(roll_max, 1e-12)
    return float(np.clip(np.max(dd) if len(dd) else 0.0, 0.0, 1.0))


def conformal_interval(
    calibration_errors: Sequence[float],
    target: float = 0.95,
) -> float:
    errors = np.asarray(calibration_errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    if len(errors) == 0:
        return 1.0
    return float(np.quantile(errors, target))


@dataclass(slots=True)
class RiskEngine:
    alpha: float = 0.05
    horizons: tuple[int, ...] = (1, 5, 20)

    def estimate(
        self,
        point_path: Sequence[float],
        interval_lower: Sequence[float],
        interval_upper: Sequence[float],
        horizon: int,
        method: str = "historical",
    ) -> RiskStats:
        k = min(int(horizon), len(point_path), len(interval_lower), len(interval_upper))
        if k <= 0:
            return RiskStats(
                horizon=horizon, var=0.0, expected_shortfall=0.0,
                expected_mdd=0.0, lower_interval=0.0, upper_interval=0.0,
            )
        path = np.asarray(point_path[:k], dtype=float)
        lo = np.asarray(interval_lower[:k], dtype=float)
        hi = np.asarray(interval_upper[:k], dtype=float)
        terminal_samples = np.sum(
            np.random.default_rng(42).normal(
                loc=path,
                scale=np.maximum((hi - lo) / (2 * 1.96), 1e-8),
                size=(2000, k),
            ),
            axis=1,
        )
        if method == "evt":
            var, es = evt_var_es(terminal_samples, self.alpha)
        else:
            var, es = historical_var_es(terminal_samples, self.alpha)
        lower = float(np.quantile(terminal_samples, self.alpha / 2.0))
        upper = float(np.quantile(terminal_samples, 1.0 - self.alpha / 2.0))
        mdd = np.mean([max_drawdown(np.cumsum(p)) for p in terminal_samples])
        return RiskStats(
            horizon=horizon,
            var=float(var),
            expected_shortfall=float(es),
            expected_mdd=float(mdd),
            lower_interval=lower,
            upper_interval=upper,
            tail_prob=self.alpha,
        )

    def build_report(
        self,
        event_date: str,
        industry_code: str,
        point_path: Sequence[float],
        lower_95: Sequence[float],
        upper_95: Sequence[float],
        method: str = "historical",
    ) -> RiskReport:
        horizons = {
            h: self.estimate(point_path, lower_95, upper_95, h, method)
            for h in self.horizons
        }
        return RiskReport(event_date=event_date, industry_code=industry_code, horizons=horizons)


def kupiec_pof(n: int, n_breach: int, alpha: float) -> float:
    if n == 0:
        return 1.0
    p = alpha
    pi = n_breach / n

    def safe(v: float) -> float:
        return min(max(v, 1e-12), 1 - 1e-12)

    ll0 = (n - n_breach) * np.log(safe(1 - p)) + n_breach * np.log(safe(p))
    ll1 = (n - n_breach) * np.log(safe(1 - pi)) + n_breach * np.log(safe(pi))
    lr = -2.0 * (ll0 - ll1)
    import math

    return float(math.erfc(math.sqrt(max(lr, 0.0) / 2.0)))


def backtest_var(realized: Sequence[float], var_pred: Sequence[float],
                 alpha: float = 0.05) -> Dict[str, float]:
    r = np.asarray(realized, dtype=float)
    v = np.asarray(var_pred, dtype=float)
    breaches = (r < v).astype(int)
    return {
        "n": int(len(r)),
        "breach_rate": float(breaches.mean()) if len(breaches) else float("nan"),
        "kupiec_p": kupiec_pof(len(r), int(breaches.sum()), alpha),
        "pinball": float(np.mean(np.where(r - v >= 0, alpha * (r - v), (alpha - 1) * (r - v))))
        if len(r) else float("nan"),
    }

