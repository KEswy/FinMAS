"""Generate a short natural-language risk commentary with DeepSeek."""
from __future__ import annotations

import json
from typing import Optional

from agents.llm_client import chat_completion


SYSTEM_PROMPT = (
    "你是一名金融科技风控中心的分析师。根据给定的风险报告摘要，"
    "用简洁、专业、可审计的中文写一段风险点评。不要虚构报告中没有的数字。"
)


def _summary(report: dict, max_items: int = 3) -> dict:
    risk = report.get("risk", {})
    return {
        "portfolio_risk": {
            str(h): risk.get("portfolio_risk", {}).get(str(h), {}).get("realized_car")
            for h in report.get("meta", {}).get("horizons", [])
        },
        "calibration": {
            str(h): risk.get("calibration", {}).get(str(h))
            for h in report.get("meta", {}).get("horizons", [])
        },
        "top_loss_events": [
            {
                "event_date": item.get("event_date"),
                "event_name": item.get("event_name"),
                "industry_code": item.get("industry_code"),
                "realized_car": item.get("realized_car_T5"),
            }
            for item in risk.get("event_risk", {}).get("5", [])[:max_items]
        ],
        "limit_warnings": risk.get("limits", {}).get("warnings", [])[:max_items],
        "stress_pnl": [
            {
                "name": item.get("name"),
                "portfolio_pnl": item.get("portfolio_pnl"),
            }
            for item in report.get("stress", {}).get("historical", [])[:max_items]
        ],
    }


def generate_commentary(
    report: dict,
    model: Optional[str] = None,
    max_attempts: int = 3,
) -> str:
    payload = json.dumps(_summary(report), ensure_ascii=False, indent=2)
    user = (
        "请基于以下风险报告摘要，写一段 150-250 字的风控点评，"
        "重点说明主要风险来源、校准情况和最值得关注的尾部风险。\n\n"
        f"{payload}"
    )
    for attempt in range(1, max_attempts + 1):
        text = chat_completion(
            system=SYSTEM_PROMPT,
            user=user,
            model=model,
            temperature=0.3 if attempt == 1 else 0.6,
            max_tokens=900,
            timeout=120,
            retries=2,
        )
        if text.strip():
            return text
    return "（DeepSeek 多次返回空结果，未能生成自动点评。）"
