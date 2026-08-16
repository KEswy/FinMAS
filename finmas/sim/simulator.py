"""Continuous market simulation runner."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from .agents import SimulationAgents
from .causal_graph import CausalGraph
from .data_source import MarketDataSource
from .schemas import CounterfactualResult, MarketTick, SimulationState, TimelineEntry


class MarketSimulator:
    def __init__(
        self,
        data_source: Optional[MarketDataSource] = None,
        agents: Optional[SimulationAgents] = None,
        causal_graph: Optional[CausalGraph] = None,
    ) -> None:
        self.data_source = data_source or MarketDataSource()
        self.agents = agents or SimulationAgents()
        self.causal_graph = causal_graph or CausalGraph()

    def _tick(self, date: str) -> MarketTick:
        return self.data_source.tick(date)

    def run(self, dates: list[str]) -> SimulationState:
        state = SimulationState()
        for date in dates:
            tick = self._tick(date)
            tick = self.data_source.detect_event(tick)
            intraday = self.data_source.intraday_snapshot(date) if tick.triggered_event else None
            opinions, paths, narrative, disagreement = self.agents.run(tick)
            for path in paths:
                self.causal_graph.add_path(path)
            state.timeline.append(
                TimelineEntry(
                    date=date,
                    tick=tick,
                    opinions=opinions,
                    causal_paths=paths,
                    narrative=narrative,
                    intraday_snapshot=intraday,
                )
            )
            state.trace.append(
                {
                    "date": date,
                    "market_return": tick.market_return,
                    "n_opinions": len(opinions),
                    "disagreement": disagreement,
                    "n_causal_paths": len(paths),
                }
            )
        return state

    def counterfactual(
        self,
        date: str,
        perturbation: dict,
    ) -> CounterfactualResult:
        tick = self._tick(date)
        original_opinions, original_paths, original, _ = self.agents.run(tick)
        modified = replace(
            tick,
            triggered_event=perturbation.get(
                "triggered_event", tick.triggered_event
            ),
            market_return=float(
                perturbation.get("market_return", tick.market_return)
            ),
        )
        counter_opinions, counter_paths, counter_narrative, _ = self.agents.run(modified)
        if self.agents.use_llm:
            try:
                counter_narrative = self.agents.provider.chat(
                    system="你是反事实解释Agent。说明如果事件不同，市场叙事会如何变化。",
                    user=(
                        f"原始市场收益{tick.market_return:+.4f}\n"
                        f"扰动后市场收益{modified.market_return:+.4f}\n"
                        f"扰动：{perturbation}"
                    ),
                    temperature=0.3,
                    use_cache=True,
                ).content.strip()
            except Exception:
                pass
        changed = [p for p in counter_paths if p not in original_paths]
        return CounterfactualResult(
            perturbation=perturbation,
            original_narrative=original,
            counterfactual_narrative=counter_narrative,
            changed_paths=changed,
        )
