"""Small, firewall-safe time-series candidate pool."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .base import TimeSeriesModel, TimeSeriesPrediction


def make_windows(x: np.ndarray, y: np.ndarray, input_len: int,
                 output_len: int, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for start in range(0, len(x) - input_len - output_len + 1, step):
        xs.append(x[start : start + input_len])
        ys.append(y[start + input_len : start + input_len + output_len])
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


class MomentumModel(TimeSeriesModel):
    name = "momentum"

    def __init__(self, horizon: int = 5, window: int = 20) -> None:
        self.horizon = horizon
        self.window = window
        self.signal = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MomentumModel":
        ret_col = x[:, 0]
        if len(ret_col) >= self.window:
            self.signal = float(np.nansum(ret_col[-self.window :]))
        return self

    def predict(self, x: np.ndarray) -> TimeSeriesPrediction:
        last_ret = float(np.nansum(x[-self.window :, 0])) if len(x) >= self.window else float(np.nansum(x[:, 0]))
        mean = np.repeat(last_ret / max(self.horizon, 1), self.horizon)
        var = np.repeat(float(np.nanstd(x[:, 0]) ** 2 + 1e-8), self.horizon)
        prob_up = 1.0 / (1.0 + np.exp(-3.0 * float(np.mean(mean))))
        return TimeSeriesPrediction(mean=mean, var=var, prob_up=prob_up,
                                    signal=float(np.mean(mean)), meta={})


class LinearTrendModel(TimeSeriesModel):
    name = "linear"

    def __init__(self, horizon: int = 5, alpha: float = 1.0) -> None:
        self.horizon = horizon
        self.alpha = alpha
        self.model = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearTrendModel":
        from sklearn.linear_model import Ridge

        xf = x.reshape(len(x), -1)
        yf = y.reshape(len(y), -1)
        self.model = Ridge(alpha=self.alpha).fit(xf, yf)
        return self

    def predict(self, x: np.ndarray) -> TimeSeriesPrediction:
        if self.model is None:
            return MomentumModel(horizon=self.horizon).predict(x)
        xf = x.reshape(1, -1)
        pred = self.model.predict(xf).reshape(-1)
        var = np.repeat(float(np.var(x[:, 0]) + 1e-8), len(pred))
        signal = float(np.mean(pred))
        prob_up = 1.0 / (1.0 + np.exp(-3.0 * signal / max(np.sqrt(float(var[0])), 1e-6)))
        return TimeSeriesPrediction(mean=pred, var=var, prob_up=prob_up,
                                    signal=signal, meta={})


class MLPModel(TimeSeriesModel):
    name = "mlp"

    def __init__(self, horizon: int = 5, hidden: int = 32, max_iter: int = 300) -> None:
        self.horizon = horizon
        self.hidden = hidden
        self.max_iter = max_iter
        self.model = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MLPModel":
        from sklearn.neural_network import MLPRegressor

        self.model = MLPRegressor(
            hidden_layer_sizes=(self.hidden,),
            max_iter=self.max_iter,
            early_stopping=True,
            random_state=42,
        ).fit(x.reshape(len(x), -1), y.reshape(len(y), -1))
        return self

    def predict(self, x: np.ndarray) -> TimeSeriesPrediction:
        if self.model is None:
            return MomentumModel(horizon=self.horizon).predict(x)
        pred = self.model.predict(x.reshape(1, -1)).reshape(-1)
        var = np.repeat(float(np.var(x[:, 0]) + 1e-8), len(pred))
        signal = float(np.mean(pred))
        prob_up = 1.0 / (1.0 + np.exp(-3.0 * signal / max(np.sqrt(float(var[0])), 1e-6)))
        return TimeSeriesPrediction(mean=pred, var=var, prob_up=prob_up,
                                    signal=signal, meta={})


class XGBoostModel(TimeSeriesModel):
    name = "xgboost"

    def __init__(self, horizon: int = 5) -> None:
        self.horizon = horizon
        self.model = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        try:
            from xgboost import XGBRegressor

            self.model = XGBRegressor(
                n_estimators=120, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
            ).fit(x.reshape(len(x), -1), y.reshape(len(y), -1))
        except ImportError:
            self.model = None
        return self

    def predict(self, x: np.ndarray) -> TimeSeriesPrediction:
        if self.model is None:
            return LinearTrendModel(horizon=self.horizon).predict(x)
        pred = self.model.predict(x.reshape(1, -1)).reshape(-1)
        var = np.repeat(float(np.var(x[:, 0]) + 1e-8), len(pred))
        signal = float(np.mean(pred))
        prob_up = 1.0 / (1.0 + np.exp(-3.0 * signal / max(np.sqrt(float(var[0])), 1e-6)))
        return TimeSeriesPrediction(mean=pred, var=var, prob_up=prob_up,
                                    signal=signal, meta={})


class ModelPool:
    """Fits several candidates and selects/ensembles them on a temporal validation split."""

    def __init__(self, horizon: int = 5, input_len: int = 60, output_len: int = 5) -> None:
        self.horizon = horizon
        self.input_len = input_len
        self.output_len = output_len
        self.models: List[TimeSeriesModel] = []
        self.weights: List[float] = []
        self._selected_name = ""

    def _candidates(self) -> List[TimeSeriesModel]:
        return [
            MomentumModel(horizon=self.output_len),
            LinearTrendModel(horizon=self.output_len),
            MLPModel(horizon=self.output_len),
            XGBoostModel(horizon=self.output_len),
        ]

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ModelPool":
        xs, ys = make_windows(x, y, self.input_len, self.output_len, step=max(self.input_len // 2, 1))
        if len(xs) < 4:
            xs = x[None, : self.input_len]
            ys = y[None, : self.output_len]
        split = max(1, int(len(xs) * 0.8))
        train_x, val_x = xs[:split], xs[split:]
        train_y, val_y = ys[:split], ys[split:]

        best = None
        best_loss = float("inf")
        fitted = []
        for model in self._candidates():
            try:
                model.fit(train_x, train_y)
                preds = np.stack([model.predict(vx).mean for vx in val_x])
                loss = float(np.mean(np.abs(preds - val_y)))
            except Exception:
                loss = float("inf")
            fitted.append((loss, model))
            if loss < best_loss:
                best_loss = loss
                best = model

        # Small inverse-loss ensemble over valid candidates.
        valid = [(loss, model) for loss, model in fitted if np.isfinite(loss)]
        if not valid:
            valid = fitted
        inv = np.array([1.0 / max(loss, 1e-6) for loss, _ in valid])
        inv = np.nan_to_num(inv, nan=0.0, posinf=0.0, neginf=0.0)
        if inv.sum() <= 0:
            inv = np.ones_like(inv)
        weights = inv / inv.sum()
        self.models = [model for _, model in valid]
        self.weights = weights.tolist()
        self._selected_name = getattr(best, "name", "unknown")
        return self

    def predict(self, x: np.ndarray) -> TimeSeriesPrediction:
        if not self.models:
            return MomentumModel(horizon=self.output_len).predict(x)
        preds = [m.predict(x) for m in self.models]
        weighted = np.average([p.mean for p in preds], axis=0, weights=self.weights)
        var = np.average([p.var for p in preds], axis=0, weights=self.weights)
        prob_up = float(np.average([p.prob_up for p in preds], weights=self.weights))
        signal = float(np.average([p.signal for p in preds], weights=self.weights))
        return TimeSeriesPrediction(
            mean=weighted,
            var=var,
            prob_up=prob_up,
            signal=signal,
            meta={"selected": self._selected_name, "n_models": len(self.models)},
        )
