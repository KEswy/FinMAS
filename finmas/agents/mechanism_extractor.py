"""Structured LLM mechanism extraction.

The extractor never returns a final trade direction. It returns factors,
evidence identifiers, transmission paths, and confidence metadata.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ..schemas import EventInput, EvidencePackage, FactorSignal
from .llm_provider import LLMProvider

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是金融事件机理提取器。你的任务不是直接预测行业涨跌，而是从事件文本和证据中提取结构化因子。

必须严格输出一个 JSON 对象，格式如下：
{
  "event_summary": "事件摘要",
  "chain": [
    {
      "step": 1,
      "layer": "social|industry|company|asset",
      "entity": "实体名",
      "reasoning": "推理说明",
      "evidence_ids": ["c001"],
      "confidence": 0.8,
      "impact_direction": "+|-|?",
      "impact_magnitude": "高|中|低|未知"
    }
  ],
  "overall_confidence": 0.8,
  "evidence_gaps": ["缺少X证据"],
  "recommendation": "一段简短综合判断"
}

只输出 JSON，不要 Markdown，不要额外解释。"""


class MechanismExtractor:
    def __init__(self, provider: Optional[LLMProvider] = None) -> None:
        self.provider = provider or LLMProvider()
        self._mag = {"高": 0.9, "中": 0.5, "低": 0.2, "未知": 0.0}

    def extract(self, event: EventInput, evidence: Optional[EvidencePackage] = None) -> Dict[str, Any]:
        evidence_text = self._evidence_text(evidence)
        user = (
            f"事件日期：{event.event_date}\n"
            f"事件类型：{event.event_type}\n"
            f"行业代码：{event.industry_code}\n"
            f"事件文本：{event.event_text}\n\n"
            f"可用证据（仅限事件日之前）：\n{evidence_text}\n"
        )
        try:
            return self.provider.chat_json(SYSTEM_PROMPT, user, temperature=0.1)
        except Exception as exc:
            logger.warning("LLM extraction failed, using text fallback: %s", exc)
            return self._fallback(event)

    def to_signals(self, event: EventInput, raw: Dict[str, Any]) -> List[FactorSignal]:
        signals: List[FactorSignal] = []
        chain = raw.get("chain", [])
        if isinstance(chain, dict):
            chain = [chain]
        for i, step in enumerate(chain or []):
            entity = str(step.get("entity") or f"step_{i + 1}")
            direction = str(step.get("impact_direction") or "?")
            if direction not in ("+", "-"):
                direction = "+"
            magnitude = float(self._mag.get(str(step.get("impact_magnitude")), 0.0))
            confidence = self._safe_float(step.get("confidence"), 0.5)
            evidence_ids = [str(x) for x in (step.get("evidence_ids") or [])]
            signals.append(
                FactorSignal(
                    name=f"llm_{i + 1}_{entity}",
                    value=max(min(magnitude, 1.0), 0.0),
                    confidence=max(min(confidence, 1.0), 0.0),
                    direction=direction,
                    source=str(step.get("layer") or "unknown"),
                    evidence_ids=evidence_ids,
                    metadata={
                        "entity": entity,
                        "reasoning": str(step.get("reasoning", ""))[:240],
                    },
                )
            )
        return signals

    def build_bundle_parts(self, event: EventInput,
                           evidence: Optional[EvidencePackage] = None) -> Dict[str, Any]:
        raw = self.extract(event, evidence)
        signals = self.to_signals(event, raw)
        return {
            "raw": raw,
            "signals": signals,
            "overall_confidence": self._safe_float(raw.get("overall_confidence"), 0.5),
            "evidence_gaps": list(raw.get("evidence_gaps") or []),
            "recommendation": str(raw.get("recommendation", "")),
        }

    @staticmethod
    def _evidence_text(package: Optional[EvidencePackage]) -> str:
        if package is None or not package.chunks:
            return "（无检索证据）"
        lines = []
        for c in package.chunks[:8]:
            lines.append(f"[{c.chunk_id}] {c.source}: {c.text[:240]}")
        return "\n".join(lines)

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            out = float(value)
        except Exception:
            return default
        return max(min(out, 1.0), 0.0)

    def _fallback(self, event: EventInput) -> Dict[str, Any]:
        return {
            "event_summary": event.event_text,
            "chain": [
                {
                    "step": 1,
                    "layer": "social",
                    "entity": event.event_type,
                    "reasoning": "fallback extraction from event text",
                    "evidence_ids": [],
                    "confidence": 0.5,
                    "impact_direction": "?",
                    "impact_magnitude": "未知",
                }
            ],
            "overall_confidence": 0.5,
            "evidence_gaps": ["LLM extraction failed"],
            "recommendation": "",
        }

