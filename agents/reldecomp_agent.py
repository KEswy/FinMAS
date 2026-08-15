"""
Relative-return decomposition intervention (v4 Phase 7).
========================================================
Tests the paper's attribution causally: the system predicts an industry's
ABSOLUTE move but is scored on the RELATIVE (vs-market) move, and conflates the
two. This skill forces the LLM to separate them in three explicit steps:

  step 1  R_ind : effect of the event on the industry's ABSOLUTE return (+/-)
  step 2  R_mkt : effect of the SAME event on the whole MARKET (CSI300) (+/-)
  step 3  sign(R_ind - R_mkt) : will the industry BEAT the market? -> the target

vs. the baseline which asks for the CAR (relative) sign directly. If forcing the
decomposition reduces the absolute-vs-relative misalignment, it supports the
mechanism and gives a fix. If it doesn't, the difficulty is asset-pricing, not
prompt comprehension. Either way is a reportable result.

Leak-safe: event text only (no label/prior/future). Gated by USE_RELDECOMP=1.
Emits a signed net_signal (direction * magnitude * confidence) like the debate.
"""
from __future__ import annotations
import os, re, json, logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)
_OLLAMA = "http://localhost:11434/api/chat"

_SYS = ("你是严谨的金融分析师。只用给定事件信息推理，不得虚构数据。"
        "务必区分『行业绝对涨跌』与『行业相对大盘(沪深300)的超额』——二者常常不同"
        "（政策利好可能让全市场都涨，此时行业未必跑赢）。")

_STEP3_JSON = ('严格输出一行JSON：{"r_ind":"+"或"-","r_mkt":"+"或"-",'
               '"beats_market":"+"或"-","confidence":0到1的小数,'
               '"magnitude":"高"或"中"或"低"}。beats_market 表示该行业'
               '未来5日相对沪深300是否跑赢(+)或跑输(-)。只输出JSON。')


def _call(model, system, user, temperature=0.3, timeout=180):
    import urllib.request
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False, "keep_alive": "60m",
        "options": {"temperature": temperature},
    }).encode("utf-8")
    for wait in (5, 10, 20, 40, 60):
        try:
            req = urllib.request.Request(_OLLAMA, data=payload,
                                         headers={"Content-Type": "application/json"})
            r = urllib.request.urlopen(req, timeout=timeout)
            txt = json.loads(r.read())["message"]["content"]
            return re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()
        except Exception as e:
            import time as _t; logger.warning(f"[reldecomp] retry: {e}"); _t.sleep(wait)
    return ""


_MAG = {"高": 0.9, "中": 0.5, "低": 0.2}


def run_reldecomp(event: str, industry: str, evidence: str, model: str,
                  temperature: float = 0.3) -> Tuple[float, Dict]:
    """Three-step relative-return decomposition. Returns (signed_signal, info)
    where signed_signal encodes the beats_market direction."""
    ev = (evidence or "").strip() or "（无检索证据）"
    ctx = (f"事件：{event}\n受影响行业：{industry}\n"
           f"可用证据（仅限事件日之前）：\n{ev}\n")
    prompt = (
        f"{ctx}\n请分三步作答：\n"
        f"第1步：该事件对【{industry}行业绝对收益】的方向影响(+/-)及理由(1句)。\n"
        f"第2步：该事件对【整体市场沪深300】的方向影响(+/-)及理由(1句)。\n"
        f"第3步：比较第1、2步——该行业未来5日能否【跑赢沪深300】？\n\n"
        f"{_STEP3_JSON}")
    raw = _call(model, _SYS, prompt, temperature)
    d = {}
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
        except Exception:
            d = {}
    bm = str(d.get("beats_market", "")).strip()
    if bm not in ("+", "-"):
        # fallback: derive from r_ind vs r_mkt if both present
        ri, rm = str(d.get("r_ind", "")), str(d.get("r_mkt", ""))
        if ri == "+" and rm == "-":
            bm = "+"
        elif ri == "-" and rm == "+":
            bm = "-"
        else:
            return 0.0, {"reldecomp": "unparsed", "raw": raw[:120]}
    try:
        conf = min(max(float(d.get("confidence", 0.5)), 0.0), 1.0)
    except Exception:
        conf = 0.5
    mag = _MAG.get(str(d.get("magnitude", "中")).strip(), 0.5)
    sign = 1.0 if bm == "+" else -1.0
    signal = sign * mag * conf
    return signal, {"reldecomp": "ok", "beats_market": bm,
                    "r_ind": d.get("r_ind"), "r_mkt": d.get("r_mkt"),
                    "confidence": round(conf, 3), "magnitude": mag,
                    "aligned_abs_mkt_differ": (d.get("r_ind") != d.get("r_mkt"))}
