"""LLM causal edge extraction with deterministic fallback."""

from __future__ import annotations

from typing import List, Optional

from ..agents.llm_provider import LLMProvider
from .schemas import CausalEdge


class CausalExtractor:
    def __init__(self, provider: Optional[LLMProvider] = None,
                 use_llm: bool = False) -> None:
        self.provider = provider or LLMProvider()
        self.use_llm = use_llm

    def extract(self, event: str, event_date: str,
                industries: Optional[List[str]] = None) -> List[CausalEdge]:
        if self.use_llm:
            edges = self._llm_extract(event, event_date, industries)
            if edges:
                return edges
        return self._fallback(event, event_date, industries)

    def _llm_extract(self, event: str, event_date: str,
                     industries: Optional[List[str]]) -> List[CausalEdge]:
        industries = industries or []
        prompt = (
            f"事件：{event}\n日期：{event_date}\n"
            f"相关行业：{industries}\n"
            "请输出JSON：{\"edges\":[{\"source\":\"事件或实体\",\"target\":\"行业或市场\","
            "\"relation\":\"causes/affects/supports/contradicts\","
            "\"confidence\":0到1,\"evidence\":\"证据编号\"}]}"
        )
        try:
            data = self.provider.chat_json(
                system="你是金融因果图抽取器。只输出JSON。",
                user=prompt,
                temperature=0.1,
                use_cache=True,
            )
        except Exception:
            return []
        out = []
        for raw in data.get("edges", []):
            if not raw.get("source") or not raw.get("target"):
                continue
            out.append(
                CausalEdge(
                    source=str(raw["source"]),
                    target=str(raw["target"]),
                    relation=str(raw.get("relation", "affects")),
                    time=event_date,
                    weight=1.0,
                    confidence=float(raw.get("confidence", 0.7)),
                    evidence_ids=[str(raw.get("evidence", "llm_causal"))],
                )
            )
        return out

    def _fallback(self, event: str, event_date: str,
                  industries: Optional[List[str]]) -> List[CausalEdge]:
        industries = industries or ["market"]
        edges = []
        if any(k in event for k in ["降息", "降准", "宽松", "LPR"]):
            edges.append(
                CausalEdge("event", "bank", "affects", event_date, 0.8, 0.8, ["rule"])
            )
        for industry in industries:
            edges.append(
                CausalEdge("event", str(industry), "affects", event_date, 0.7, 0.7, ["rule"])
            )
        return edges

