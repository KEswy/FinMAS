"""End-to-end FinMAS v5 orchestration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .agents.llm_provider import LLMProvider
from .agents.mechanism_extractor import MechanismExtractor
from .data.feature_store import FeatureStore
from .data.time_firewall import TimeFirewall
from .fusion.decision import DirectionFusion
from .models.base import TimeSeriesPrediction
from .models.pool import ModelPool, make_windows
from .rag.knowledge_graph import KnowledgeGraph
from .rag.retrieval import build_retriever
from .risk.engine import RiskEngine
from .schemas import DirectionDecision, EventInput, EvidencePackage, SignalBundle

logger = logging.getLogger(__name__)


class FinMASPipeline:
    def __init__(
        self,
        data_dir: str = "data",
        llm_provider: Optional[LLMProvider] = None,
        use_llm: bool = True,
        horizon: int = 5,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.horizon = int(horizon)
        self.store = FeatureStore(data_dir=str(self.data_dir))
        self.provider = llm_provider or LLMProvider()
        self.use_llm = use_llm
        self.extractor = MechanismExtractor(self.provider)
        self.retriever = build_retriever(str(self.data_dir / "events" / "doc_library.json"))
        self.kg = KnowledgeGraph()
        self.kg.build_from_event_library(str(self.data_dir / "events" / "event_library.csv"))
        self.model_pool = ModelPool(horizon=self.horizon, input_len=60, output_len=self.horizon)
        self.fusion = DirectionFusion()
        self.risk = RiskEngine(horizons=(1, 5, 20))

    def retrieve_evidence(self, event: EventInput) -> EvidencePackage:
        package = self.retriever.retrieve(
            query=f"{event.event_text} {event.industry_code} {event.event_type}",
            top_k=8,
            as_of_date=event.event_date,
        )
        if not package.chunks:
            package = EvidencePackage(query=package.query, chunks=[])
        return package

    def _kg_summary(self, event: EventInput, evidence: EvidencePackage) -> str:
        entities = [str(event.industry_code), event.event_type]
        for chunk in evidence.chunks:
            entities.extend(chunk.entities)
        sub = self.kg.get_subgraph(entities[:20])
        return sub.summary()

    def _irf_prior(self, event: EventInput) -> Dict[str, float]:
        path = self.data_dir / "irf_priors.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cell = data.get("priors", {}).get(event.event_type, {}).get(str(event.industry_code), {})
            direction = int(cell.get("direction", 0))
            strength = float(cell.get("strength", 0.0))
            return {"direction": direction, "strength": strength,
                    "signed_strength": direction * strength}
        except Exception:
            return {}

    def _time_series_signal(self, event: EventInput) -> TimeSeriesPrediction:
        matrix = self.store.pre_event_matrix(event.industry_code, event.event_date, lookback=600)
        if matrix is None or len(matrix) < self.model_pool.input_len + self.model_pool.output_len + 5:
            # Cold start fallback: use last available returns as a flat path.
            ret = np.asarray([0.0] * self.horizon, dtype=float)
            var = np.asarray([0.0001] * self.horizon, dtype=float)
            return TimeSeriesPrediction(mean=ret, var=var, prob_up=0.5,
                                        signal=0.0, meta={"cold_start": True})

        x = matrix.to_numpy(dtype=float)
        y = x[:, 0:1]  # target is next-step return for training windows
        try:
            self.model_pool.fit(x, y)
        except Exception as exc:
            logger.warning("time-series fit failed: %s", exc)
        last = x[-self.model_pool.input_len :]
        return self.model_pool.predict(last)

    def _training_rows(self, event: EventInput) -> tuple[np.ndarray, np.ndarray, List[str]]:
        car_path = self.data_dir / "processed" / "car_results_expanded.csv"
        if not car_path.exists():
            car_path = self.data_dir / "processed" / "car_results.csv"
        car = pd.read_csv(car_path)
        car = car[car["window"] == 5].copy()
        car["event_date"] = car["event_date"].astype(str).str[:10]
        car["industry_code"] = car["industry_code"].astype(str)
        event_dt = pd.Timestamp(event.event_date)
        past = car[pd.to_datetime(car["event_date"]) < event_dt].drop_duplicates(
            ["event_date", "industry_code"]
        )
        rows: List[List[float]] = []
        labels: List[int] = []
        types: List[str] = []
        for _, row in past.iterrows():
            try:
                features = self.store.build_features(
                    row["industry_code"], row["event_date"],
                    include_text=False,
                )
                ts_prob = self._simple_momentum_prob(row["industry_code"], row["event_date"])
                irf = self._irf_prior(
                    EventInput(
                        event_date=row["event_date"],
                        event_text="",
                        event_type=str(row.get("event_type", "")),
                        industry_code=row["industry_code"],
                    )
                )
                label = int(row["CAR"] >= 0)
                vec = [
                    0.0,
                    features.get("froth", 0.0),
                    irf.get("signed_strength", 0.0),
                    0.0,
                    0.0,
                    ts_prob,
                    features.get("momentum_20", 0.0),
                    features.get("valuation_signal", 0.0),
                ]
                rows.append(vec)
                labels.append(label)
                types.append(str(row.get("event_type", "")))
            except Exception:
                continue
        return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int), types

    def _simple_momentum_prob(self, industry_code: str, event_date: str) -> float:
        try:
            df = self.store.pre_event_matrix(industry_code, event_date, lookback=60)
            if df is None or df.empty:
                return 0.5
            ret = df["ret"].to_numpy(dtype=float)
            signal = float(np.nansum(ret[-20:])) if len(ret) >= 20 else float(np.nansum(ret))
            return float(1.0 / (1.0 + np.exp(-3.0 * signal / 0.02)))
        except Exception:
            return 0.5

    def _fit_fusion(self, event: EventInput) -> None:
        x, y, types = self._training_rows(event)
        if len(x) >= 10:
            self.fusion.fit_from_past(x, y, types)

    def predict(self, event: EventInput) -> DirectionDecision:
        evidence = self.retrieve_evidence(event)
        evidence.subgraph_summary = self._kg_summary(event, evidence)

        llm_parts = {"signals": [], "overall_confidence": 0.5, "evidence_gaps": [], "recommendation": ""}
        if self.use_llm:
            llm_parts = self.extractor.build_bundle_parts(event, evidence)

        ts = self._time_series_signal(event)
        valuation = self.store.valuation_snapshot(event.industry_code, event.event_date)
        sentiment = self.store.sentiment_snapshot(event.event_date)
        irf = self._irf_prior(event)
        disagreement = self._disagreement(llm_parts["signals"], ts)

        bundle = SignalBundle(
            event=event,
            llm_factors=llm_parts["signals"],
            time_series=ts.to_dict(),
            valuation=valuation,
            sentiment=sentiment,
            irf_prior=irf,
            disagreement=disagreement,
            evidence=evidence,
        )
        self._fit_fusion(event)
        decision = self.fusion.decide(bundle, horizon=self.horizon)
        decision.evidence_ids = [c.chunk_id for c in evidence.chunks[:10]]
        if llm_parts.get("recommendation"):
            decision.reason = llm_parts["recommendation"][:240]

        # Attach risk only for committed events.
        ts_mean = ts.mean
        ts_var = np.maximum(ts.var, 1e-8)
        z = 1.96
        lo = ts_mean - z * np.sqrt(ts_var)
        hi = ts_mean + z * np.sqrt(ts_var)
        decision.risk = self.risk.estimate(ts_mean, lo, hi, self.horizon)
        return decision

    @staticmethod
    def _disagreement(llm_factors, ts_prediction: TimeSeriesPrediction) -> float:
        if not llm_factors:
            return 0.0
        llm_signal = float(np.sum([f.signed_value() for f in llm_factors]))
        ts_signal = float(ts_prediction.signal)
        diff = abs(np.tanh(llm_signal) - np.tanh(ts_signal * 5.0))
        return float(min(max(diff, 0.0), 1.0))
