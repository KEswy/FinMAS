"""Switchable risk-measure estimators.

All functions operate on a return-like series where a negative value is a loss.
VaR is reported as the lower alpha-quantile; ES is the average below VaR.
"""
from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np


def _as_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def historical_var_es(returns: Sequence[float], alpha: float) -> Dict[str, float]:
    arr = _as_array(returns)
    if len(arr) == 0:
        return {"var": float("nan"), "es": float("nan")}
    var = float(np.quantile(arr, alpha))
    tail = arr[arr <= var]
    es = float(np.mean(tail)) if len(tail) else var
    return {"var": var, "es": es}


def bootstrap_var_es(
    returns: Sequence[float],
    alpha: float,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    arr = _as_array(returns)
    if len(arr) == 0:
        return {"var": float("nan"), "es": float("nan")}
    rng = np.random.default_rng(seed)
    n = len(arr)
    var_samples = []
    es_samples = []
    for _ in range(max(int(n_bootstrap), 0)):
        idx = rng.integers(0, n, n)
        sample = arr[idx]
        var = float(np.quantile(sample, alpha))
        tail = sample[sample <= var]
        var_samples.append(var)
        es_samples.append(float(np.mean(tail)) if len(tail) else var)
    return {
        "var": float(np.mean(var_samples)) if var_samples else historical_var_es(arr, alpha)["var"],
        "es": float(np.mean(es_samples)) if es_samples else historical_var_es(arr, alpha)["es"],
    }


def parametric_var_es(returns: Sequence[float], alpha: float) -> Dict[str, float]:
    arr = _as_array(returns)
    if len(arr) < 2:
        return historical_var_es(arr, alpha)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    z = float(np.quantile(np.random.default_rng(0).normal(size=1000000), alpha))
    var = mu + z * sigma
    es = mu - sigma * (np.exp(-(z**2) / 2.0) / (alpha * np.sqrt(2.0 * np.pi)))
    return {"var": var, "es": float(es)}


def ewma_var_es(
    returns: Sequence[float],
    alpha: float,
    lam: float = 0.94,
) -> Dict[str, float]:
    arr = _as_array(returns)
    if len(arr) < 2:
        return historical_var_es(arr, alpha)
    mu = float(np.mean(arr))
    squared = (arr - mu) ** 2
    weights = np.power(lam, np.arange(len(squared) - 1, -1, -1))
    sigma = float(np.sqrt(np.sum(squared * weights) / np.sum(weights)))
    z = float(np.quantile(np.random.default_rng(0).normal(size=1000000), alpha))
    var = mu + z * sigma
    es = mu - sigma * (np.exp(-(z**2) / 2.0) / (alpha * np.sqrt(2.0 * np.pi)))
    return {"var": var, "es": float(es)}


def evt_var_es(
    returns: Sequence[float],
    alpha: float,
    tail_frac: float = 0.10,
) -> Dict[str, float]:
    """Method-of-moments peak-over-threshold GPD estimator.

    The estimator is fitted on losses (``-returns``) and transformed back to
    signed return space, so VaR and ES remain negative tail-risk values.
    """
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
    nu = float(len(losses))
    nu_threshold = float(len(excess))
    if abs(shape) < 1e-6:
        loss_var = threshold - scale * np.log((nu / nu_threshold) * alpha)
        loss_es = loss_var + scale
    else:
        tail_prob = (nu / nu_threshold) * alpha
        loss_var = threshold + (scale / shape) * ((tail_prob ** (-shape)) - 1.0)
        loss_es = (loss_var + scale - shape * threshold) / (1.0 - shape)
    return {"var": float(-loss_var), "es": float(-loss_es)}


def estimate_risk(
    returns: Sequence[float],
    alpha: float,
    method: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    method = str(method).lower()
    if method == "historical":
        return historical_var_es(returns, alpha)
    if method == "bootstrap":
        return bootstrap_var_es(returns, alpha, n_bootstrap, seed)
    if method == "parametric":
        return parametric_var_es(returns, alpha)
    if method == "ewma":
        return ewma_var_es(returns, alpha)
    if method == "evt":
        return evt_var_es(returns, alpha)
    raise ValueError(f"unsupported risk method: {method}")


def estimate_risk_grid(
    returns: Sequence[float],
    alphas: Iterable[float],
    methods: Iterable[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for alpha in alphas:
        out[str(alpha)] = {
            str(method): estimate_risk(returns, float(alpha), str(method), n_bootstrap, seed)
            for method in methods
        }
    return out
