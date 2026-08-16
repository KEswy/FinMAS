"""Observation agents for continuous market simulation."""

from __future__ import annotations

from typing import List, Optional

from ..agents.llm_provider import LLMProvider
from .schemas import AgentOpinion, CausalPath, MarketTick


class SimulationAgents:
    def __init__(self, provider: Optional[LLMProvider] = None) -> None:
        self.provider = provider or LLMProvider()

    def observer(self, tick: MarketTick) -> AgentOpinion:
        confidence = min(max(abs(tick.market_return) * 20.0, 0.2), 0.95)
        return AgentOpinion(
            agent="Observer",
            opinion=(
                f"市场当日收益 {tick.market_return:+.4f}，"
                f"资金流变化 {tick.capital_flow.get('margin_change', 0.0):+.4f}"
            ),
            confidence=confidence,
            evidence_ids=["market_tick"],
        )

    def technician(self, tick: MarketTick) -> AgentOpinion:
        top = sorted(tick.industry_returns.items(), key=lambda x: -x[1])[:3]
        bottom = sorted(tick.industry_returns.items(), key=lambda x: x[1])[:3]
        return AgentOpinion(
            agent="Technician",
            opinion=f"强势行业: {top}; 弱势行业: {bottom}",
            confidence=0.65,
            evidence_ids=["industry_returns"],
        )

    def fundamental(self, tick: MarketTick) -> AgentOpinion:
        return AgentOpinion(
            agent="Fundamental",
            opinion="从估值与宏观状态看，市场短期方向仍取决于政策和资金流持续性。",
            confidence=0.55,
            evidence_ids=["valuation_proxy"],
        )

    def news(self, tick: MarketTick) -> AgentOpinion:
        if tick.triggered_event:
            opinion = f"触发事件：{tick.triggered_event}"
            confidence = 0.8
        else:
            opinion = "未检测到新的重大事件。"
            confidence = 0.3
        return AgentOpinion(
            agent="News",
            opinion=opinion,
            confidence=confidence,
            evidence_ids=[tick.triggered_event] if tick.triggered_event else [],
        )

    def causal(self, tick: MarketTick) -> List[CausalPath]:
        paths = [
            CausalPath("market", "all_industries", "affects", 0.8, ["market_tick"]),
            CausalPath("capital_flow", "market", "causes", 0.6, ["margin_change"]),
        ]
        if tick.triggered_event:
            paths.append(
                CausalPath("triggered_event", "market", "causes", 0.75, [tick.triggered_event])
            )
        return paths

    def storyteller(self, tick: MarketTick, opinions: List[AgentOpinion]) -> str:
        parts = [o.opinion for o in opinions if o.confidence >= 0.5]
        event = tick.triggered_event or "无明显事件"
        return f"今日{event}。市场收益{tick.market_return:+.4f}。{' '.join(parts[:3])}"

    def judge(self, opinions: List[AgentOpinion]) -> float:
        if not opinions:
            return 0.0
        positive = sum(1 for o in opinions if any(k in o.opinion for k in ["强", "上涨", "利好"]))
        negative = sum(1 for o in opinions if any(k in o.opinion for k in ["弱", "下跌", "利空"]))
        return min(max(abs(positive - negative) / max(len(opinions), 1), 0.0), 1.0)

    def run(self, tick: MarketTick) -> tuple[List[AgentOpinion], List[CausalPath], str, float]:
        opinions = [
            self.observer(tick),
            self.technician(tick),
            self.fundamental(tick),
            self.news(tick),
        ]
        paths = self.causal(tick)
        narrative = self.storyteller(tick, opinions)
        disagreement = self.judge(opinions)
        return opinions, paths, narrative, disagreement

