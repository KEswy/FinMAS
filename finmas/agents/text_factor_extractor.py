"""Deterministic text-factor fallback for environments without an LLM.

The fallback mirrors the structured-factor contract of MechanismExtractor:
it emits FactorSignal objects, not a final direction.
"""

from __future__ import annotations

from typing import List

from ..schemas import EventInput, FactorSignal


POS_TERMS = {
    "降息", "降准", "宽松", "减息", "下调", "一揽子", "组合拳",
    "减免", "补贴", "扶持", "活跃", "刺激", "提振", "托底",
    "松绑", "放开", "支持", "突破", "超预期", "利好", "降至",
    "减半", "里程碑", "降税", "扩围", "扩容", "注册制",
}

NEG_TERMS = {
    "关税", "制裁", "加征", "报复", "反制", "暴跌", "大跌",
    "崩盘", "熔断", "黑色", "双减", "整顿", "打压", "处罚",
    "罚款", "违约", "退市", "爆雷", "破产", "债务", "收紧",
    "禁止", "限制", "冻结", "利空", "风险", "下滑", "萎缩",
    "衰退", "承压", "重挫", "拖累",
}


class TextFactorExtractor:
    """Heuristic extraction of signed event factors from event text."""

    def extract(self, event: EventInput) -> List[FactorSignal]:
        text = str(event.event_text)
        pos = sum(1 for term in POS_TERMS if term in text)
        neg = sum(1 for term in NEG_TERMS if term in text)
        if pos == 0 and neg == 0:
            return []

        direction = "+" if pos > neg else "-" if neg > pos else "?"
        magnitude = 0.9 if max(pos, neg) >= 3 else 0.55 if max(pos, neg) >= 2 else 0.30
        confidence = 0.70 if direction != "?" else 0.40
        signals = []

        if pos:
            signals.append(
                FactorSignal(
                    name="text_positive",
                    value=magnitude,
                    confidence=confidence,
                    direction="+",
                    source="keyword",
                    metadata={"hits": pos},
                )
            )
        if neg:
            signals.append(
                FactorSignal(
                    name="text_negative",
                    value=magnitude,
                    confidence=confidence,
                    direction="-",
                    source="keyword",
                    metadata={"hits": neg},
                )
            )
        if direction == "?":
            signals.append(
                FactorSignal(
                    name="text_mixed",
                    value=0.2,
                    confidence=0.35,
                    direction="+",
                    source="keyword",
                    metadata={"pos": pos, "neg": neg},
                )
            )
        return signals

