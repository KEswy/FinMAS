"""
Temporal Firewall (v4 Phase 0)
==============================

The single choke-point that enforces walk-forward information isolation for
every event prediction. For an event on date ``t``, NOTHING computed after ``t``
may influence the prediction. This module closes the three leak channels the
v3 audit found (the price/windowing path was already firewalled in v3):

  L1  Sensitivity prior computed over the FULL CAR archive (incl. the target
      event itself and future events).  -> ``sensitivity_prior(event_dt)``
      filters ``event_date < t`` before calling ``compute_sensitivity_v2``.

  L2  ``direction`` / ``magnitude`` label read from event_library.csv at
      prediction time.  -> ``label_guard`` is *always on*: walk-forward never
      reads the outcome-adjacent label. (``_infer_direction_from_text`` on the
      event *description* stays legal — it is model input, not the answer.)

  L3/L4/L5  RAG had ZERO temporal filtering: docs published T+1/T+2 (inside the
      CAR window) and undated mechanism docs were retrievable by every event.
      -> ``FirewalledRetriever`` filters chunks to ``pub_time < t`` and re-ranks
      age relative to ``t`` (not wall-clock ``time.time()``).

Design principle: NON-INVASIVE. We wrap the copied-from-v3 ``Retriever`` and
``compute_sensitivity_v2`` rather than editing their internals, so the v4
foundation stays byte-identical to v3 and the firewall is the only new surface.
The accuracy delta is therefore attributable purely to leakage removal.

Usage
-----
    from firewall import WalkForwardContext
    ctx = WalkForwardContext(event_date="2024-07-22")
    prior   = ctx.sensitivity_prior()                 # dict {(etype,ind): s}
    package = ctx.retrieve(query, retriever, entity_names=...)
    ctx.assert_no_future_leakage(package)             # hard check
"""
from __future__ import annotations

import os
import time
import datetime as _dt
from dataclasses import dataclass, field
from typing import Optional, List

import pandas as pd


# ──────────────────────────────────────────────────────────────
# Date helpers
# ──────────────────────────────────────────────────────────────
def _to_date(x) -> _dt.date:
    """Coerce str/datetime/date/pandas.Timestamp to a plain date."""
    if isinstance(x, _dt.date) and not isinstance(x, _dt.datetime):
        return x
    if isinstance(x, _dt.datetime):
        return x.date()
    if isinstance(x, pd.Timestamp):
        return x.date()
    # string forms like "2024-07-22" or "2024-07-22 00:00:00"
    return _dt.datetime.strptime(str(x)[:10], "%Y-%m-%d").date()


def _to_unix(d: _dt.date) -> float:
    """Midnight of date `d` as a unix timestamp (local), matching how
    doc pub_time is built in retrieval.build_from_library (time.mktime of
    strptime %Y-%m-%d, i.e. local midnight)."""
    return time.mktime(_dt.datetime(d.year, d.month, d.day).timetuple())


# ──────────────────────────────────────────────────────────────
# Firewalled retriever wrapper
# ──────────────────────────────────────────────────────────────
class FirewalledRetriever:
    """A drop-in replacement for a v3 ``Retriever`` that only ever sees docs
    which existed before the event date.

    Instead of mutating the source retriever's private index in place (fragile:
    BM25Index keeps ``_tf``/``_df``/``_avgdl`` index-aligned with ``_chunks``,
    so slicing one list corrupts the others), we build ONE clean index over the
    visible-chunk subset at construction time, via the source retriever's own
    ``index_documents`` path. The mechanism agent calls ``retrieve`` once per
    CoT step (5×/event); all those calls reuse this single pre-filtered index,
    so we rebuild per event, not per step.
    """

    STRICT = "strict"        # pub_time < t  (default; excludes event-day docs)
    INCLUSIVE = "inclusive"  # pub_time <= t (event-day docs allowed)

    def __init__(self, retriever, event_dt: _dt.date, mode: str = STRICT,
                 drop_undated: bool = True):
        self.event_dt = event_dt
        self.cutoff_ts = _to_unix(event_dt)
        self.mode = mode
        # undated docs (no real pub_date; pub_time fell back to time.time())
        # are quarantined by default: an undated "timeless" doc that in fact
        # cites forward aggregates is a leak we cannot verify, so exclude it
        # unless it carries an explicit valid_from <= t (see _visible).
        self.drop_undated = drop_undated

        all_chunks = self._source_chunks(retriever)
        self.visible = [c for c in all_chunks if self._visible(c)]
        self.hidden = [c for c in all_chunks if not self._visible(c)]

        # Build a fresh retriever of the SAME class over only the visible set,
        # reusing the source's KG. index_documents() rebuilds _tf/_df/_avgdl
        # and the faiss cache correctly for the subset.
        self._r = retriever.__class__(kg=getattr(retriever, "kg", None))
        if self.visible:
            self._r.index_documents(list(self.visible))

    def _visible(self, chunk) -> bool:
        """True if `chunk` legally existed at event_dt."""
        # explicit valid_from in metadata wins (used by the rewritten,
        # genuinely time-invariant mechanism_kb; each doc declares when it
        # became knowable).
        meta = getattr(chunk, "metadata", None) or {}
        vf = meta.get("valid_from")
        if vf:
            try:
                return _to_date(vf) <= self.event_dt
            except Exception:
                pass
        pt = getattr(chunk, "pub_time", None)
        if pt is None:
            return not self.drop_undated
        # A doc with a real pub_date but flagged undated-origin is dropped too.
        if meta.get("undated_origin") and self.drop_undated:
            return False
        if self.mode == self.INCLUSIVE:
            return pt <= self.cutoff_ts
        return pt < self.cutoff_ts

    def retrieve(self, query, entity_names=None, step_context="") -> "object":
        """Firewalled retrieve. Signature-compatible with Retriever.retrieve.

        The underlying index already contains only visible docs; we additionally
        pin the age-decay clock to the event date so rerank favors docs recent
        *as of t*, not as of wall-clock 2026.
        """
        with _event_relative_clock(self.cutoff_ts):
            return self._r.retrieve(query, entity_names=entity_names,
                                    step_context=step_context)

    @staticmethod
    def _source_chunks(retriever) -> list:
        """Pull the full chunk list from the source retriever's vector index
        (BM25 holds the same chunk objects; either is a complete inventory)."""
        vi = getattr(retriever, "vector_index", None)
        if vi is not None and hasattr(vi, "_chunks") and vi._chunks:
            return list(vi._chunks)
        bi = getattr(retriever, "bm25_index", None)
        if bi is not None and hasattr(bi, "_chunks"):
            return list(bi._chunks)
        return []

    def assert_no_future_leakage(self, package) -> None:
        """Hard assertion: every chunk in the package must be firewall-visible.

        Reuses ``_visible`` (the single source of truth) rather than re-checking
        pub_time, so a legitimately time-invariant doc (valid_from <= t) whose
        pub_time happens to post-date t is NOT flagged. A chunk that is not
        ``_visible`` yet appears in the package is a genuine leak.
        """
        for c in getattr(package, "chunks", []):
            if not self._visible(c):
                pt = getattr(c, "pub_time", None)
                pt_str = (time.strftime('%Y-%m-%d', time.localtime(pt))
                          if pt is not None else "undated")
                raise AssertionError(
                    f"LEAKAGE: chunk {getattr(c,'chunk_id','?')} "
                    f"pub_time={pt_str} not visible at event_date={self.event_dt}"
                )


# ──────────────────────────────────────────────────────────────
# Event-relative clock: makes EvidenceChunk.age_days() measure age
# relative to the event date, not wall-clock now. Patches time.time
# for the retrieval module only, for the duration of one retrieve call.
# ──────────────────────────────────────────────────────────────
class _event_relative_clock:
    def __init__(self, as_of_ts: float):
        self.as_of_ts = as_of_ts
        self._orig = None

    def __enter__(self):
        import rag.retrieval as _ret
        self._mod = _ret
        self._orig = _ret.time.time
        as_of = self.as_of_ts
        # age_days = (time.time() - pub_time)/86400  -> pin time.time to event
        self._mod.time.time = lambda: as_of
        return self

    def __exit__(self, *exc):
        if self._orig is not None:
            self._mod.time.time = self._orig
        return False


# ──────────────────────────────────────────────────────────────
# Walk-forward context: the object threaded through run_analysis
# ──────────────────────────────────────────────────────────────
@dataclass
class WalkForwardContext:
    """Owns all time-isolation services for predicting one event at date t."""
    event_date: str
    car_csv: str = field(
        default_factory=lambda: os.environ.get(
            "CAR_CSV", "data/processed/car_results.csv"))
    mode: str = FirewalledRetriever.STRICT
    n_full: int = 5
    n_pool_min: int = 2
    _event_dt: _dt.date = field(init=False)

    def __post_init__(self):
        self._event_dt = _to_date(self.event_date)

    # -- L1: leak-free sensitivity prior --------------------------
    def sensitivity_prior(self, df5: Optional[pd.DataFrame] = None) -> dict:
        """Compute the sensitivity matrix using ONLY events strictly before t.

        Generalizes the LOO/OOT monkey-patch (run_all_events_v2_loo.py:34,
        run_oot_2025.py:51) into a first-class walk-forward call. Replaces the
        hardcoded full-archive SECTOR_SENSITIVITY dict.
        """
        from scripts.calibrate_sensitivity_v2 import compute_sensitivity_v2
        if df5 is None:
            df = pd.read_csv(self.car_csv)
            df5 = df[df["window"] == 5].copy()
            df5["industry_code"] = df5["industry_code"].astype(str)
        # strict walk-forward: prior sees only PAST events
        past = df5[df5["event_date"].apply(lambda d: _to_date(d) < self._event_dt)]
        if len(past) == 0:
            return {}   # cold start: no history -> everything falls back to s=1.0
        return compute_sensitivity_v2(past, n_full=self.n_full,
                                      n_pool_min=self.n_pool_min)

    def n_past_events(self, df5: Optional[pd.DataFrame] = None) -> int:
        """How many events the prior can learn from (cold-start diagnostics)."""
        if df5 is None:
            df = pd.read_csv(self.car_csv)
            df5 = df[df["window"] == 5]
        dates = df5["event_date"].apply(_to_date)
        return int((dates < self._event_dt).sum())

    # -- L3/L4/L5: firewalled retrieval ---------------------------
    def wrap_retriever(self, retriever) -> FirewalledRetriever:
        return FirewalledRetriever(retriever, self._event_dt, mode=self.mode)

    def retrieve(self, query, retriever, entity_names=None, step_context=""):
        fr = self.wrap_retriever(retriever)
        pkg = fr.retrieve(query, entity_names=entity_names, step_context=step_context)
        fr.assert_no_future_leakage(pkg)
        return pkg

    # -- L2: label guard ------------------------------------------
    @property
    def allow_label(self) -> bool:
        """Walk-forward NEVER reads the outcome-adjacent direction label."""
        return False
