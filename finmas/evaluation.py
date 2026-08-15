"""Walk-forward evaluation utilities for the calibrated decision layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data.feature_store import FeatureStore
from .fusion.calibration import TemporalCalibrator
from .agents.llm_provider import LLMProvider
from .agents.mechanism_extractor import MechanismExtractor
from .agents.text_factor_extractor import TextFactorExtractor
from .schemas import EventInput


FEATURE_ORDER = [
    "disagreement",
    "froth",
    "irf_prior",
    "llm_bull_bear",
    "llm_net",
    "ts_prob_up",
    "ts_signal",
    "valuation_signal",
]


def load_irf(data_dir: str = "data") -> Dict[str, Dict[str, Dict[str, float]]]:
    path = Path(data_dir) / "irf_priors.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8")).get("priors", {})
        out: Dict[str, Dict[str, Dict[str, float]]] = {}
        for event_type, cells in raw.items():
            out[str(event_type)] = {
                str(code): {
                    "direction": float(cell.get("direction", 0)),
                    "strength": float(cell.get("strength", 0)),
                    "signed_strength": float(cell.get("direction", 0))
                    * float(cell.get("strength", 0)),
                }
                for code, cell in cells.items()
            }
        return out
    except Exception:
        return {}


class WalkForwardEvaluator:
    def __init__(
        self,
        data_dir: str = "data",
        car_csv: str = "data/processed/car_results_expanded.csv",
        llm_mode: str = "none",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.store = FeatureStore(data_dir=str(self.data_dir))
        self.irf = load_irf(data_dir)
        self.llm_mode = llm_mode.lower()
        self.llm_provider = LLMProvider() if self.llm_mode == "llm" else None
        self.llm_extractor = MechanismExtractor(self.llm_provider) if self.llm_provider else None
        self.text_extractor = TextFactorExtractor() if self.llm_mode == "heuristic" else None
        self.car = pd.read_csv(car_csv)
        self.car = self.car[self.car["window"] == 5].copy()
        self.car["event_date"] = self.car["event_date"].astype(str).str[:10]
        self.car["industry_code"] = self.car["industry_code"].astype(str)

    def rows(self) -> pd.DataFrame:
        df = self.car.drop_duplicates(["event_date", "industry_code"]).copy()
        df["_dt"] = pd.to_datetime(df["event_date"])
        df = df.sort_values("_dt").reset_index(drop=True)
        return df

    @staticmethod
    def _momentum_prob(df: pd.DataFrame, window: int = 20) -> float:
        if df is None or df.empty or "ret" not in df.columns:
            return 0.5
        ret = df["ret"].to_numpy(dtype=float)
        if len(ret) < window:
            signal = float(np.nansum(ret))
        else:
            signal = float(np.nansum(ret[-window:]))
        scale = float(np.nanstd(ret) + 1e-8)
        return float(1.0 / (1.0 + np.exp(-3.0 * signal / max(scale * 5.0, 0.01))))

    def _llm_stats(self, factors) -> tuple[float, float, float]:
        if not factors:
            return 0.0, 0.0, 0.0
        bull = float(sum(max(f.signed_value(), 0.0) for f in factors))
        bear = float(sum(max(-f.signed_value(), 0.0) for f in factors))
        net = bull - bear
        total = abs(bull) + abs(bear)
        disagreement = 0.0 if total < 1e-8 else min(1.0, 2.0 * min(bull, bear) / total)
        return net, net, disagreement

    def _extract_llm_factors(self, event_date: str, event_type: str,
                             industry_code: str, event_text: str):
        if self.llm_mode == "none":
            return []
        event = EventInput(
            event_date=event_date,
            event_text=event_text,
            event_type=event_type,
            industry_code=industry_code,
        )
        if self.llm_mode == "heuristic" and self.text_extractor:
            return self.text_extractor.extract(event)
        if self.llm_mode == "llm" and self.llm_extractor:
            try:
                return self.llm_extractor.to_signals(
                    event, self.llm_extractor.extract(event)
                )
            except Exception:
                return []
        return []

    def build_vector(self, event_date: str, event_type: str, industry_code: str,
                     event_text: str = "") -> Optional[np.ndarray]:
        features = self.store.build_features(
            industry_code, event_date, include_text=False
        )
        matrix = self.store.pre_event_matrix(industry_code, event_date, lookback=120)
        ts_prob = self._momentum_prob(matrix)
        irf = self.irf.get(event_type, {}).get(str(industry_code), {})
        llm_net, llm_bull_bear, disagreement = self._llm_stats(
            self._extract_llm_factors(event_date, event_type, industry_code, event_text)
        )
        values = {
            "disagreement": disagreement,
            "froth": float(features.get("froth", 0.0) or 0.0),
            "irf_prior": float(irf.get("signed_strength", 0.0) or 0.0),
            "llm_bull_bear": llm_bull_bear,
            "llm_net": llm_net,
            "ts_prob_up": float(ts_prob or 0.5),
            "ts_signal": float(features.get("momentum_20", 0.0) or 0.0),
            "valuation_signal": float(features.get("valuation_signal", 0.0) or 0.0),
        }
        return np.asarray([values[name] for name in FEATURE_ORDER], dtype=float)

    def run(self, limit: Optional[int] = None, min_train: int = 30) -> pd.DataFrame:
        rows = self.rows()
        if limit:
            rows = rows.head(int(limit))

        history_x: List[np.ndarray] = []
        history_y: List[int] = []
        history_types: List[str] = []
        results: List[Dict[str, object]] = []

        for _, row in rows.iterrows():
            vec = self.build_vector(
                row["event_date"],
                row["event_type"],
                row["industry_code"],
                str(row.get("event_name", "")),
            )
            label = int(row["CAR"] >= 0)
            if vec is None:
                continue
            if len(history_x) >= min_train:
                cal = TemporalCalibrator()
                cal.fit(np.asarray(history_x, dtype=float), np.asarray(history_y, dtype=int))
                prob = float(cal.predict_proba(vec.reshape(1, -1))[0])
            else:
                prob = self._raw_score(vec)

            threshold = self._selective_threshold(history_types, history_y, row["event_type"])
            confidence = abs(prob - 0.5) * 2.0
            abstain = confidence < threshold
            direction = "+" if prob >= 0.5 else "-"
            results.append(
                {
                    "event_date": row["event_date"],
                    "event_type": row["event_type"],
                    "industry_code": row["industry_code"],
                    "real_dir": "+" if label == 1 else "-",
                    "final_dir": direction,
                    "prob_up": prob,
                    "confidence": confidence,
                    "abstain": abstain,
                    "dir_correct": bool((direction == "+") == bool(label)),
                    "committed_dir_correct": bool(
                        (not abstain) and ((direction == "+") == bool(label))
                    ),
                    "real_CAR": float(row["CAR"]),
                }
            )
            history_x.append(vec)
            history_y.append(label)
            history_types.append(str(row["event_type"]))
        return pd.DataFrame(results)

    @staticmethod
    def _raw_score(vec: np.ndarray) -> float:
        ts_prob = float(vec[5])
        valuation = float(vec[7])
        froth = float(vec[1])
        irf = float(vec[2])
        score = (
            0.55 * ts_prob
            + 0.10 * np.tanh(valuation)
            + 0.10 * np.tanh(froth)
            + 0.10 * np.tanh(irf)
        )
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _selective_threshold(types: List[str], labels: List[int], event_type: str) -> float:
        if len(types) < 10:
            return 0.12
        arr = np.asarray(types, dtype=str)
        mask = arr == event_type
        if int(mask.sum()) < 5:
            return 0.12
        y = np.asarray(labels, dtype=int)[mask]
        # A simple abstention threshold: require confidence above the observed imbalance.
        return float(min(max(abs(y.mean() - 0.5) * 2.0, 0.10), 0.25))
