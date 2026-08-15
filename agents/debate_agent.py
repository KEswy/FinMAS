"""
Debate agent (v4 Phase 6) -- genuine multi-agent interaction.
=============================================================
Answers the reviewer's "this is a modular pipeline, not a multi-agent system":
a BULL agent and a BEAR agent exchange messages over N rounds (each sees and
rebuts the other's latest argument -> real message passing), then a JUDGE agent
reads the full transcript and resolves the conflict into a signed direction,
confidence, and magnitude (real conflict resolution).

Leak-safety: bull/bear/judge see ONLY the event description (legitimate input)
and the firewalled retrieved evidence (pub_date < t). No labels, no future data.
Same discipline as MechanismAgent -- the debate adds interaction, not information.

Gated by USE_DEBATE=1 (default off -> pipeline byte-identical to before).
Output feeds the same net_signal correction pipeline in main.py.
"""
from __future__ import annotations
import os, re, json, logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

N_ROUNDS = int(os.environ.get("DEBATE_ROUNDS", "3"))     # bull+bear exchanges
_OLLAMA = "http://localhost:11434/api/chat"


def _call(model: str, system: str, user: str, temperature: float = 0.4,
          timeout: int = 180) -> str:
    """Single Ollama chat call, <think> stripped. Mirrors MechanismAgent._call_llm
    (same endpoint/model) but standalone so the debate module is self-contained."""
    import urllib.request, urllib.error
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False, "keep_alive": "60m",
        "options": {"temperature": temperature},
    }).encode("utf-8")
    backoff = [5, 10, 20, 40, 60]
    for wait in backoff:
        try:
            req = urllib.request.Request(_OLLAMA, data=payload,
                                         headers={"Content-Type": "application/json"})
            r = urllib.request.urlopen(req, timeout=timeout)
            txt = json.loads(r.read())["message"]["content"]
            return re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()
        except Exception as e:
            import time as _t; logger.warning(f"[debate] retry: {e}"); _t.sleep(wait)
    return ""


_BULL_SYS = ("你是看多分析师。基于给定事件与证据，论证该行业在事件后5日相对沪深300"
             "会上涨。必须针对对方(看空方)的最新论点逐条反驳。只用给定证据，"
             "不得虚构数据。3-5句，中文。")
_BEAR_SYS = ("你是看空分析师。基于给定事件与证据，论证该行业在事件后5日相对沪深300"
             "会下跌。必须针对对方(看多方)的最新论点逐条反驳。只用给定证据，"
             "不得虚构数据。3-5句，中文。")
_JUDGE_SYS = ("你是中立裁判。阅读看多/看空的完整辩论记录，判定哪方更有说服力，"
              "给出最终方向。同时诚实评估这是否是一个『势均力敌、难以判断』的"
              "争议事件——如果双方论据都有道理、证据不足以明确定论，contested 记为 true。"
              "严格输出一行JSON："
              '{"direction":"+"或"-","confidence":0到1的小数,'
              '"contested":true或false,'
              '"magnitude":"高"或"中"或"低","reason":"一句话"}。只输出JSON。')


def _ctx(event: str, industry: str, evidence: str) -> str:
    ev = evidence.strip() or "（无检索证据）"
    return f"事件：{event}\n受影响行业：{industry}\n可用证据（仅限事件日之前）：\n{ev}\n"


def run_debate(event: str, industry: str, evidence: str, model: str,
               n_rounds: int = N_ROUNDS, temperature: float = 0.4,
               abstain_threshold: float = None
               ) -> Tuple[float, Dict]:
    """Bull vs bear over n_rounds with message passing, then a judge (variant B:
    confidence-weighted with ABSTENTION). Returns (signed_signal, info).

    signed_signal is on the net_signal scale = direction * magnitude * confidence,
    BUT if the judge's confidence is below abstain_threshold (or the judge itself
    abstains), signed_signal is 0.0 and info['abstained']=True. Abstention lets
    the system commit only on high-consensus events (selective prediction), scored
    as a coverage-vs-accuracy trade-off rather than forced on every scenario.

    abstain_threshold defaults to env DEBATE_ABSTAIN (else 0.6)."""
    if abstain_threshold is None:
        abstain_threshold = float(os.environ.get("DEBATE_ABSTAIN", "0.6"))
    ctx = _ctx(event, industry, evidence)
    transcript: List[Dict[str, str]] = []
    bull_last = bear_last = ""
    for rnd in range(1, n_rounds + 1):
        bull_user = (f"{ctx}\n【本轮第{rnd}轮】对方(看空)最新论点："
                     f"{bear_last or '（尚无，请先开场陈述看多逻辑）'}\n请陈述/反驳：")
        bull_last = _call(model, _BULL_SYS, bull_user, temperature)
        transcript.append({"role": "bull", "round": rnd, "text": bull_last})
        bear_user = (f"{ctx}\n【本轮第{rnd}轮】对方(看多)最新论点：{bull_last}\n"
                     f"请针对性反驳并陈述看空逻辑：")
        bear_last = _call(model, _BEAR_SYS, bear_user, temperature)
        transcript.append({"role": "bear", "round": rnd, "text": bear_last})

    convo = "\n".join(f"[{t['role']}·R{t['round']}] {t['text']}" for t in transcript)
    verdict_raw = _call(model, _JUDGE_SYS, f"{ctx}\n辩论记录：\n{convo}\n\n请裁决(仅JSON)：",
                        temperature=0.2)
    signal, info = _parse_verdict(verdict_raw)

    # ── variant B: abstain on CONTESTED events, decoupled from the judge's
    # self-reported confidence (miscalibrated: fired 319/321 in the first run).
    # The judge flags a close call explicitly; abstain if contested OR low conf. ──
    conf = info.get("confidence", 0.0)
    contested = _parse_bool(info.get("contested"))
    abstained = (info.get("debate") != "ok") or contested or (conf < abstain_threshold)
    info.update({"n_rounds": n_rounds, "abstain_threshold": abstain_threshold,
                 "contested": contested, "abstained": bool(abstained),
                 "committed_signal": signal,
                 "transcript": transcript, "verdict_raw": verdict_raw[:200]})
    if abstained:
        return 0.0, info                    # no correction -> defers to time-series
    return signal, info


def _parse_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


_MAG = {"高": 0.9, "中": 0.5, "低": 0.2}


def _parse_verdict(raw: str) -> Tuple[float, Dict]:
    """Parse judge JSON -> (signed_signal, info). Robust to code fences / prose.
    Falls back to keyword scan; abstains (0.0) only if truly unparseable."""
    d = {}
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
        except Exception:
            d = {}
    direction = str(d.get("direction", "")).strip()
    if direction not in ("+", "-"):
        head = raw[:60]
        if ("+" in head or "涨" in head or "利好" in head) and "跌" not in head:
            direction = "+"
        elif "-" in head or "跌" in head or "利空" in head:
            direction = "-"
        else:
            return 0.0, {"debate": "unparsed"}
    try:
        conf = float(d.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    conf = min(max(conf, 0.0), 1.0)
    mag = _MAG.get(str(d.get("magnitude", "中")).strip(), 0.5)
    sign = 1.0 if direction == "+" else -1.0
    signal = sign * mag * conf
    return signal, {"debate": "ok", "direction": direction, "confidence": round(conf, 3),
                    "magnitude": mag, "contested": d.get("contested", False),
                    "reason": str(d.get("reason", ""))[:80]}


def evidence_text_from_package(pkg) -> str:
    """Extract a short evidence string from a firewalled EvidencePackage."""
    try:
        chunks = getattr(pkg, "chunks", []) or []
        parts = []
        for c in chunks[:5]:
            t = getattr(c, "text", None) or getattr(c, "content", "")
            if t:
                parts.append(str(t)[:200])
        return "\n".join(f"- {p}" for p in parts)
    except Exception:
        return ""
