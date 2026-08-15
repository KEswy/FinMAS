"""
Risk-control objective (v4 Phase 4)
===================================

Risk is a FIRST-CLASS objective here, not a byproduct of direction accuracy.
Even if CAR-sign accuracy is near chance, a calibrated event-risk forecast
(how bad can [t, t+k] get, and how confident are we) is independently useful
and independently evaluated.

Two parts:
  1. RiskHead — turns ONE event's forecast artifacts (point path, conformal
     interval, KF/predictive variance, cross-industry dispersion) into an
     event-level risk block: event_VaR, expected_MDD, tail_prob, risk_interval.
     Distribution-free (uses the conformal interval directly); no EVT/parametric
     tail fit — the overfitting guard for n=168 × pred_len=24 tiny tail samples.

  2. Backtesting — evaluated SEPARATELY from direction:
       - Kupiec POF (unconditional coverage) LR test
       - Christoffersen independence LR test (are VaR breaches clustered?)
       - pinball (quantile) loss for the VaR quantile
       - drawdown calibration (predicted vs realized MDD)
     Pooled across events (per-event tails are too thin alone).

All inputs are decimal returns, consistent with the rest of the pipeline.
Realized risk labels are used ONLY for evaluation, never as model input at t.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import math

import numpy as np


# ──────────────────────────────────────────────────────────────
# Per-event risk head
# ──────────────────────────────────────────────────────────────
@dataclass
class EventRisk:
    event_VaR: float            # alpha-quantile of the [t,t+k] cumulative-return dist (loss, signed)
    expected_MDD: float         # expected max drawdown over the predicted path [0,1]
    tail_prob: float            # model's prob that realized CAR breaches the lower interval
    risk_interval: tuple        # (lower, upper) predictive interval for terminal CAR
    cvar_95: float = 0.0        # expected shortfall below event_VaR
    prob_breach_lower95: float = 0.0
    n_paths: int = 0
    method: str = "monte_carlo"
    legacy_event_VaR: float = 0.0
    legacy_expected_MDD: float = 0.0
    dispersion: float = 0.0     # cross-industry disagreement feature (from TransmissionMap)
    alpha: float = 0.05

    def to_dict(self) -> dict:
        return {
            "event_VaR": round(self.event_VaR, 5),
            "expected_MDD": round(self.expected_MDD, 5),
            "tail_prob": round(self.tail_prob, 5),
            "risk_interval": [round(self.risk_interval[0], 5),
                              round(self.risk_interval[1], 5)],
            "cvar_95": round(self.cvar_95, 5),
            "prob_breach_lower95": round(self.prob_breach_lower95, 5),
            "n_paths": self.n_paths,
            "method": self.method,
            "legacy_event_VaR": round(self.legacy_event_VaR, 5),
            "legacy_expected_MDD": round(self.legacy_expected_MDD, 5),
            "dispersion": round(self.dispersion, 5),
            "alpha": self.alpha,
        }


class RiskHead:
    """Distribution-free event-risk from forecast artifacts."""

    def __init__(self, alpha: float = 0.05, n_paths: int = 2000,
                 seed: int = 42, method: str = "monte_carlo"):
        self.alpha = alpha
        self.n_paths = int(n_paths)
        self.seed = int(seed)
        self.method = method

    @staticmethod
    def _path_max_drawdown(cum_return_path: np.ndarray) -> float:
        if len(cum_return_path) == 0:
            return 0.0
        nav = 1.0 + np.asarray(cum_return_path, dtype=float)
        roll_max = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
        dd = (roll_max - nav) / np.maximum(roll_max, 1e-12)
        return float(np.clip(np.max(dd) if len(dd) else 0.0, 0.0, 1.0))

    def _simulate_terminal_risk(self, point: np.ndarray, lo: np.ndarray,
                                hi: np.ndarray, dispersion: float):
        z95 = 1.96
        sigma = np.abs(hi - lo) / (2.0 * z95)
        dispersion_scale = 1.0 + max(float(dispersion), 0.0)
        sigma = sigma * dispersion_scale
        sigma = np.maximum(sigma, 0.0)

        rng = np.random.default_rng(self.seed)
        if self.n_paths <= 0:
            return np.asarray([point], dtype=float)
        if float(np.max(sigma)) <= 0.0:
            return np.tile(point, (self.n_paths, 1))
        shocks = rng.normal(
            loc=np.broadcast_to(point, (self.n_paths, len(point))),
            scale=np.broadcast_to(sigma, (self.n_paths, len(sigma))),
        )
        return shocks

    @staticmethod
    def _legacy_risk(point: np.ndarray, lo: np.ndarray,
                     hi: np.ndarray, alpha: float, dispersion: float):
        cum_lo = float(np.sum(lo))
        cum_hi = float(np.sum(hi))
        cum_path = np.cumsum(point)
        nav = 1.0 + cum_path
        roll_max = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
        dd = (roll_max - nav) / (roll_max + 1e-8)
        legacy_mdd = float(np.clip(np.max(dd) if len(dd) else 0.0, 0, 1))
        legacy_tail = float(min(0.5, alpha * (1.0 + max(dispersion, 0.0))))
        return cum_lo, cum_hi, legacy_mdd, legacy_tail

    def compute(self,
                point_path: Sequence[float],
                lower_95: Sequence[float],
                upper_95: Sequence[float],
                dispersion: float = 0.0,
                k: int = 5) -> EventRisk:
        """point_path/lower/upper are per-step real-scale return forecasts.
        We aggregate to the terminal [t, t+k] cumulative-return (CAR) view.

        event_VaR: the lower conformal bound on cumulative CAR at horizon k
                   (a distribution-free α-quantile proxy — the interval is
                   conformal-calibrated, so its lower edge is the risk floor).
        expected_MDD: max drawdown of the predicted cumulative path.
        tail_prob: widened by cross-industry dispersion (more disagreement →
                   fatter tail → higher breach probability), capped at 0.5.
        """
        k_eff = min(int(k), len(point_path), len(lower_95), len(upper_95))
        if k_eff <= 0:
            return EventRisk(
                event_VaR=0.0, expected_MDD=0.0, tail_prob=0.0,
                risk_interval=(0.0, 0.0), cvar_95=0.0,
                prob_breach_lower95=0.0, n_paths=0,
                method=self.method, dispersion=float(dispersion),
                alpha=self.alpha,
            )

        p = np.asarray(point_path[:k_eff], dtype=float)
        lo = np.asarray(lower_95[:k_eff], dtype=float)
        hi = np.asarray(upper_95[:k_eff], dtype=float)
        legacy_lo, legacy_hi, legacy_mdd, legacy_tail = self._legacy_risk(
            p, lo, hi, self.alpha, dispersion)

        if self.method == "legacy":
            return EventRisk(
                event_VaR=legacy_lo,
                expected_MDD=legacy_mdd,
                tail_prob=legacy_tail,
                risk_interval=(legacy_lo, legacy_hi),
                cvar_95=legacy_lo,
                prob_breach_lower95=0.0,
                n_paths=0,
                method="legacy",
                legacy_event_VaR=legacy_lo,
                legacy_expected_MDD=legacy_mdd,
                dispersion=float(dispersion),
                alpha=self.alpha,
            )

        paths = self._simulate_terminal_risk(p, lo, hi, dispersion)
        terminal_returns = np.sum(paths, axis=1)
        event_var = float(np.quantile(terminal_returns, self.alpha))
        tail_mask = terminal_returns <= event_var
        cvar = float(np.mean(terminal_returns[tail_mask])) if np.any(tail_mask) else event_var
        lower_q = float(np.quantile(terminal_returns, self.alpha / 2.0))
        upper_q = float(np.quantile(terminal_returns, 1.0 - self.alpha / 2.0))

        path_mdds = [self._path_max_drawdown(np.cumsum(path)) for path in paths]
        expected_mdd = float(np.mean(path_mdds)) if path_mdds else legacy_mdd
        prob_breach_lower95 = float(np.mean(terminal_returns <= legacy_lo))
        tail_prob = float(min(
            0.5,
            max(prob_breach_lower95,
                self.alpha * (1.0 + max(float(dispersion), 0.0)))
        ))

        return EventRisk(
            event_VaR=event_var,
            expected_MDD=expected_mdd,
            tail_prob=tail_prob,
            risk_interval=(lower_q, upper_q),
            cvar_95=cvar,
            prob_breach_lower95=prob_breach_lower95,
            n_paths=self.n_paths,
            method=self.method,
            legacy_event_VaR=legacy_lo,
            legacy_expected_MDD=legacy_mdd,
            dispersion=float(dispersion),
            alpha=self.alpha,
        )


# ──────────────────────────────────────────────────────────────
# VaR backtesting (evaluated separately from direction accuracy)
# ──────────────────────────────────────────────────────────────
@dataclass
class VaRBacktest:
    n: int
    n_breach: int
    breach_rate: float
    expected_rate: float
    kupiec_LR: float
    kupiec_p: float
    christoffersen_LR: float
    christoffersen_p: float
    pinball_loss: float
    mdd_mae: float = float("nan")     # |predicted MDD − realized MDD| mean
    passes_kupiec: bool = False

    def to_dict(self) -> dict:
        return {k: (round(v, 5) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def _chi2_sf_1df(x: float) -> float:
    """Survival function of chi-square with 1 dof (p-value), no scipy."""
    if x <= 0:
        return 1.0
    # P(X>x) = erfc(sqrt(x/2)) for 1 dof
    return math.erfc(math.sqrt(x / 2.0))


def _chi2_sf(x: float, dof: int) -> float:
    if dof == 1:
        return _chi2_sf_1df(x)
    if dof == 2:
        return math.exp(-x / 2.0)
    # fallback: Wilson–Hilferty approximation → normal
    t = ((x / dof) ** (1.0 / 3.0) - (1 - 2.0 / (9 * dof))) / math.sqrt(2.0 / (9 * dof))
    return 0.5 * math.erfc(t / math.sqrt(2))


def kupiec_pof(n: int, n_breach: int, alpha: float) -> tuple:
    """Kupiec proportion-of-failures unconditional-coverage LR test."""
    if n == 0:
        return 0.0, 1.0
    pi = n_breach / n
    p = alpha
    # avoid log(0)
    def _safe(a):
        return max(min(a, 1 - 1e-12), 1e-12)
    ll_null = (n - n_breach) * math.log(_safe(1 - p)) + n_breach * math.log(_safe(p))
    ll_alt = (n - n_breach) * math.log(_safe(1 - pi)) + n_breach * math.log(_safe(pi))
    lr = -2.0 * (ll_null - ll_alt)
    return float(lr), float(_chi2_sf_1df(lr))


def christoffersen_independence(breaches: Sequence[int]) -> tuple:
    """LR test that VaR breaches are independent (not clustered)."""
    b = list(int(x) for x in breaches)
    if len(b) < 2:
        return 0.0, 1.0
    n00 = n01 = n10 = n11 = 0
    for prev, cur in zip(b[:-1], b[1:]):
        if prev == 0 and cur == 0: n00 += 1
        elif prev == 0 and cur == 1: n01 += 1
        elif prev == 1 and cur == 0: n10 += 1
        else: n11 += 1
    def _safe(a):
        return max(min(a, 1 - 1e-12), 1e-12)
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    if pi in (0.0, 1.0):
        return 0.0, 1.0
    ll_null = (n00 + n10) * math.log(_safe(1 - pi)) + (n01 + n11) * math.log(_safe(pi))
    ll_alt = (n00 * math.log(_safe(1 - pi01)) + n01 * math.log(_safe(pi01)) +
              n10 * math.log(_safe(1 - pi11)) + n11 * math.log(_safe(pi11)))
    lr = -2.0 * (ll_null - ll_alt)
    return float(lr), float(_chi2_sf_1df(lr))


def pinball_loss(realized: Sequence[float], var_pred: Sequence[float],
                 alpha: float) -> float:
    """Quantile (pinball) loss for the α-quantile VaR forecast."""
    r = np.asarray(realized, dtype=float)
    q = np.asarray(var_pred, dtype=float)
    diff = r - q
    loss = np.where(diff >= 0, alpha * diff, (alpha - 1.0) * diff)
    return float(np.mean(loss)) if len(loss) else float("nan")


def backtest_var(realized_car: Sequence[float],
                 var_pred: Sequence[float],
                 alpha: float = 0.05,
                 realized_mdd: Optional[Sequence[float]] = None,
                 pred_mdd: Optional[Sequence[float]] = None) -> VaRBacktest:
    """Pooled VaR backtest across events. A 'breach' = realized CAR below the
    predicted VaR (lower bound). Direction accuracy plays NO role here."""
    r = np.asarray(realized_car, dtype=float)
    v = np.asarray(var_pred, dtype=float)
    n = len(r)
    breaches = (r < v).astype(int)
    n_breach = int(breaches.sum())
    breach_rate = float(n_breach / n) if n else 0.0

    k_lr, k_p = kupiec_pof(n, n_breach, alpha)
    c_lr, c_p = christoffersen_independence(breaches)
    pin = pinball_loss(r, v, alpha)

    mdd_mae = float("nan")
    if realized_mdd is not None and pred_mdd is not None and len(realized_mdd):
        mdd_mae = float(np.mean(np.abs(np.asarray(realized_mdd) -
                                       np.asarray(pred_mdd))))

    return VaRBacktest(
        n=n, n_breach=n_breach, breach_rate=breach_rate, expected_rate=alpha,
        kupiec_LR=k_lr, kupiec_p=k_p,
        christoffersen_LR=c_lr, christoffersen_p=c_p,
        pinball_loss=pin, mdd_mae=mdd_mae,
        passes_kupiec=bool(k_p > 0.05),
    )
