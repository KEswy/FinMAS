"""LangGraph node functions."""

from __future__ import annotations

import time
from typing import Any, Dict

from ..agents.llm_provider import LLMProvider
from ..schemas import FactorSignal, SignalBundle
from .state import DebateVerdict, GraphState


def _record(state: GraphState, name: str, status: str, elapsed: float,
            summary: str = "") -> None:
    state.trace.append(
        {
            "node": name,
            "status": status,
            "elapsed": round(elapsed, 6),
            "summary": summary,
        }
    )


def retrieval_node(state: GraphState, pipeline) -> GraphState:
    start = time.time()
    try:
        state.evidence = pipeline.retrieve_evidence(state.event)
        _record(state, "retrieval", "ok", time.time() - start,
                f"chunks={len(state.evidence.chunks)}")
    except Exception as exc:
        _record(state, "retrieval", "error", time.time() - start, str(exc))
    return state


def mechanism_node(state: GraphState, pipeline) -> GraphState:
    start = time.time()
    if not pipeline.use_llm:
        _record(state, "mechanism", "skipped", time.time() - start, "use_llm=false")
        return state
    try:
        parts = pipeline.extractor.build_bundle_parts(state.event, state.evidence)
        state.mechanism_signals = parts["signals"]
        _record(state, "mechanism", "ok", time.time() - start,
                f"factors={len(state.mechanism_signals)}")
    except Exception as exc:
        _record(state, "mechanism", "error", time.time() - start, str(exc))
    return state


def _debate_rounds(provider: LLMProvider, event_text: str, rounds: int) -> Dict[str, Any]:
    bull_system = "你是看多分析师，用中文3-5句论证该行业未来5日相对沪深300会跑赢。"
    bear_system = "你是看空分析师，用中文3-5句论证该行业未来5日相对沪深300会跑输。"
    judge_system = (
        '你是中立裁判。只输出JSON：{"direction":"+"或"-","confidence":0到1,'
        '"contested":true或false,"magnitude":"高"或"中"或"低","reason":"一句话"}'
    )
    bull = bear = ""
    transcript = []
    for rnd in range(1, rounds + 1):
        bull = provider.chat(
            bull_system,
            f"事件：{event_text}\n对方看空观点：{bear or '无'}\n第{rnd}轮看多论证。",
            temperature=0.4,
        ).content
        bear = provider.chat(
            bear_system,
            f"事件：{event_text}\n对方看多观点：{bull}\n第{rnd}轮看空反驳。",
            temperature=0.4,
        ).content
        transcript.append({"round": str(rnd), "bull": bull, "bear": bear})
    raw = provider.chat_json(
        judge_system,
        f"事件：{event_text}\n辩论记录：{transcript}",
        temperature=0.1,
    )
    return {"raw": raw, "transcript": transcript}


def debate_node(state: GraphState, pipeline, rounds: int = 2) -> GraphState:
    start = time.time()
    if not pipeline.use_llm:
        _record(state, "debate", "skipped", time.time() - start, "use_llm=false")
        return state
    try:
        result = _debate_rounds(pipeline.provider, state.event.event_text, rounds)
        raw = result.get("raw", {})
        direction = str(raw.get("direction", "?"))
        if direction not in ("+", "-"):
            direction = "?"
        confidence = float(raw.get("confidence", 0.5))
        magnitude = float({"高": 0.9, "中": 0.5, "低": 0.2}.get(raw.get("magnitude"), 0.5))
        contested = bool(raw.get("contested", False))
        evidence_ids = [c.chunk_id for c in state.evidence.chunks[:8]] if state.evidence else []
        signals = [
            FactorSignal(
                name="debate_consensus",
                value=magnitude,
                confidence=confidence,
                direction=direction if direction in ("+", "-") else "+",
                source="judge",
                evidence_ids=evidence_ids,
                metadata={"contested": contested},
            )
        ]
        state.debate_transcript = result.get("transcript", [])
        state.judge_verdict = DebateVerdict(
            consensus_factors=signals,
            disagreement=0.50 if contested else min(abs(confidence - 0.5) * 1.2, 0.40),
            contested=contested,
            evidence_ids=evidence_ids,
            rounds=rounds,
        )
        _record(state, "debate", "ok", time.time() - start,
                f"rounds={rounds} contested={contested}")
    except Exception as exc:
        state.judge_verdict = DebateVerdict(
            disagreement=0.9, contested=True, rounds=rounds
        )
        _record(state, "debate", "error", time.time() - start, str(exc))
    return state


def time_series_node(state: GraphState, pipeline) -> GraphState:
    start = time.time()
    try:
        pred = pipeline._time_series_signal(state.event)
        state.time_series = pred.to_dict()
        _record(state, "time_series", "ok", time.time() - start,
                f"prob_up={pred.prob_up:.3f}")
    except Exception as exc:
        _record(state, "time_series", "error", time.time() - start, str(exc))
    return state


def fusion_node(state: GraphState, pipeline) -> GraphState:
    start = time.time()
    all_factors = list(state.mechanism_signals)
    if state.judge_verdict:
        all_factors.extend(state.judge_verdict.consensus_factors)
    state.valuation = pipeline.store.valuation_snapshot(
        state.event.industry_code, state.event.event_date
    )
    state.sentiment = pipeline.store.sentiment_snapshot(state.event.event_date)
    state.irf_prior = pipeline._irf_prior(state.event)
    disagreement = max(state.disagreement, state.judge_verdict.disagreement
                       if state.judge_verdict else 0.0)
    bundle = SignalBundle(
        event=state.event,
        llm_factors=all_factors,
        time_series=state.time_series,
        valuation=state.valuation,
        sentiment=state.sentiment,
        irf_prior=state.irf_prior,
        disagreement=disagreement,
        evidence=state.evidence,
    )
    pipeline._fit_fusion(state.event)
    state.decision = pipeline.fusion.decide(bundle, horizon=pipeline.horizon)
    _record(state, "fusion", "ok", time.time() - start,
            f"prob_up={state.decision.prob_up:.3f} abstain={state.decision.abstain}")
    return state


def risk_node(state: GraphState, pipeline) -> GraphState:
    start = time.time()
    if state.decision and state.time_series:
        import numpy as np

        mean = np.asarray(state.time_series.get("mean", [0.0] * pipeline.horizon))
        var = np.maximum(np.asarray(state.time_series.get("var", [1e-8] * pipeline.horizon)), 1e-8)
        z = 1.96
        lo = mean - z * np.sqrt(var)
        hi = mean + z * np.sqrt(var)
        risk = pipeline.risk.estimate(mean, lo, hi, pipeline.horizon)
        state.risk = risk.to_dict()
        state.decision.risk = risk
        _record(state, "risk", "ok", time.time() - start,
                f"var={risk.var:.5f}")
    else:
        _record(state, "risk", "skipped", time.time() - start)
    return state
