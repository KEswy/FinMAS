"""Common time-series model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(slots=True)
class TimeSeriesPrediction:
    mean: np.ndarray
    var: np.ndarray
    prob_up: float
    signal: float
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "prob_up": float(self.prob_up),
            "signal": float(self.signal),
            "meta": self.meta,
        }


class TimeSeriesModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> "TimeSeriesModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray) -> TimeSeriesPrediction:
        raise NotImplementedError

