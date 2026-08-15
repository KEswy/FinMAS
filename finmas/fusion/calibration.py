"""Small-sample, time-ordered probability calibration."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class TemporalCalibrator:
    """Calibrate a raw score to a probability using a temporal validation split."""

    def __init__(self, method: str = "isotonic", max_train: int = 500) -> None:
        self.method = method
        self.max_train = int(max_train)
        self._model = None
        self._coef: Optional[np.ndarray] = None
        self._intercept: float = 0.0
        self.feature_names: list[str] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TemporalCalibrator":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if len(x) > self.max_train:
            x = x[-self.max_train :]
            y = y[-self.max_train :]
        if len(x) == 0:
            self._model = None
            return self

        split = max(1, int(len(x) * 0.7))
        train_x, val_x = x[:split], x[split:]
        train_y, val_y = y[:split], y[split:]

        if self.method == "isotonic" and len(train_x) >= 10 and train_x.shape[1] == 1:
            try:
                from sklearn.isotonic import IsotonicRegression

                self._model = IsotonicRegression(
                    y_min=0.05, y_max=0.95, out_of_bounds="clip"
                ).fit(train_x[:, 0], train_y)
                return self
            except Exception:
                pass

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
        model.fit(train_x, train_y)
        self._model = model
        self._coef = model.coef_[0]
        self._intercept = float(model.intercept_[0])
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if self._model is None:
            return np.full(len(x), 0.5, dtype=float)
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(x)[:, 1]
        return self._model.predict(x)
