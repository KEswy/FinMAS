"""Time-series candidate model interfaces."""

from .base import TimeSeriesModel, TimeSeriesPrediction
from .pool import ModelPool

__all__ = ["TimeSeriesModel", "TimeSeriesPrediction", "ModelPool"]
