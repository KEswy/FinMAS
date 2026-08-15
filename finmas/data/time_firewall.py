"""Small, explicit temporal firewall helpers used by the new data layer."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from ..schemas import EvidenceChunk, EvidencePackage


def to_date(value: Any) -> _dt.date:
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    return _dt.datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


@dataclass(slots=True)
class TimeFirewall:
    event_date: str
    event_dt: _dt.date = field(init=False)

    def __post_init__(self) -> None:
        self.event_dt = to_date(self.event_date)

    def rows_visible(self, frame: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        dates = pd.to_datetime(frame[date_col])
        return frame[dates.dt.date < self.event_dt].copy()

    def chunk_visible(self, chunk: EvidenceChunk) -> bool:
        meta = chunk.metadata or {}
        if meta.get("valid_from"):
            try:
                return to_date(meta["valid_from"]) <= self.event_dt
            except Exception:
                pass
        return to_date(pd.Timestamp(chunk.pub_time, unit="s").date()) < self.event_dt

    def filter_evidence(self, package: EvidencePackage) -> EvidencePackage:
        visible = [c for c in package.chunks if self.chunk_visible(c)]
        return EvidencePackage(
            query=package.query,
            chunks=visible,
            subgraph_summary=package.subgraph_summary,
            total_tokens_est=package.total_tokens_est,
            passed_time_firewall=True,
        )
