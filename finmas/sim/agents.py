"""Observation agents for continuous market simulation."""

from __future__ import annotations

from typing import List, Optional

from ..agents.llm_provider import LLMProvider
from .schemas import AgentOpinion, CausalPath, MarketTick


class SimulationAgents:
    def __init__(self, provider: Optional[LLMProvider] = None,
                 use_llm: bool = False) -> None:
        self.provider = provider or LLMProvider()
        self.use_llm = use_llm

    def _llm_opinion(self, agent: str, prompt: str) -> str:
        system = (
            f"你是金融多智能体系统中的{agent}。只输出3-5句中文观点，"
            "必须基于给定市场快照，不要编造具体数字。"
        )
        try:
            return self.provider.chat(
                system=system,
                user=prompt,
                temperature=0.3,
                use_cache=True,
            ).content.strip()
        except Exception:
            return prompt

    def observer(self, tick: MarketTick) -> AgentOpinion:
        if self.use_llm:
            text = self._llm_opinion(
                "Observer",
                f"市场收益{tick.market_return:+.4f}，资金流{tick.capital_flow}",
            )
            return AgentOpinion("Observer", text, 0.65, ["llm_observer"])
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
        if self.use_llm:
            text = self._llm_opinion(
                "Technician",
                f"行业收益：{tick.industry_returns}",
            )
            return AgentOpinion("Technician", text, 0.60, ["llm_technician"])
        top = sorted(tick.industry_returns.items(), key=lambda x: -x[1])[:3]
        bottom = sorted(tick.industry_returns.items(), key=lambda x: x[1])[:3]
        return AgentOpinion(
            agent="Technician",
            opinion=f"强势行业: {top}; 弱势行业: {bottom}",
            confidence=0.65,
            evidence_ids=["industry_returns"],
        )

    def fundamental(self, tick: MarketTick) -> AgentOpinion:
        if self.use_llm:
            text = self._llm_opinion(
                "Fundamental",
                f"市场收益{tick.market_return:+.4f}，资金流{tick.capital_flow}",
            )
            return AgentOpinion("Fundamental", text, 0.55, ["llm_fundamental"])
        return AgentOpinion(
            agent="Fundamental",
            opinion="从估值与宏观状态看，市场短期方向仍取决于政策和资金流持续性。",
            confidence=0.55,
            evidence_ids=["valuation_proxy"],
        )

    def news(self, tick: MarketTick) -> AgentOpinion:
        if self.use_llm:
            text = self._llm_opinion(
                "News",
                f"触发事件：{tick.triggered_event or '无'}",
            )
            return AgentOpinion("News", text, 0.70, ["llm_news"])
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
        top_industries = sorted(
            tick.industry_returns.items(), key=lambda x: -abs(x[1])
        )[:3]
        for industry, _ in top_industries:
            paths.append(
                CausalPath("event", industry, "affects", 0.7, ["industry_move"])
            )
        if tick.triggered_event:
            paths.append(
                CausalPath("triggered_event", "market", "causes", 0.75, [tick.triggered_event])
            )
        return paths

    def storyteller(self, tick: MarketTick, opinions: List[AgentOpinion]) -> str:
        if self.use_llm:
            parts = "\n".join(f"{o.agent}: {o.opinion}" for o in opinions)
            try:
                return self.provider.chat(
                    system="你是金融叙事Storyteller。用3-5句中文整合多Agent观点，形成连贯叙事。",
                    user=f"市场收益{tick.market_return:+.4f}\n{parts}",
                    temperature=0.3,
                    use_cache=True,
                ).content.strip()
            except Exception:
                pass
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
