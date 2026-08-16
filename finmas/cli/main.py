"""FinMAS v5 command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..pipeline import FinMASPipeline
from ..report import build_markdown_report
from ..schemas import EventInput


def main() -> None:
    parser = argparse.ArgumentParser(description="FinMAS v5")
    sub = parser.add_subparsers(dest="command", required=True)

    predict = sub.add_parser("predict", help="predict one event/industry direction")
    predict.add_argument("--event-date", required=True)
    predict.add_argument("--event-text", required=True)
    predict.add_argument("--event-type", required=True)
    predict.add_argument("--industry-code", required=True)
    predict.add_argument("--no-llm", action="store_true")

    serve = sub.add_parser("serve", help="start FastAPI server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    report = sub.add_parser("report", help="build a consolidated Markdown report")
    report.add_argument("--result-csv", required=True)
    report.add_argument("--risk-json", default=None)
    report.add_argument("--portfolio-json", default=None)
    report.add_argument("--output-md", default=None)

    args = parser.parse_args()
    if args.command == "predict":
        event = EventInput(
            event_date=args.event_date,
            event_text=args.event_text,
            event_type=args.event_type,
            industry_code=args.industry_code,
        )
        pipeline = FinMASPipeline(use_llm=not args.no_llm)
        decision = pipeline.predict(event)
        print(json.dumps(decision.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "serve":
        import uvicorn

        uvicorn.run("finmas.api.app:app", host=args.host, port=args.port)
        return

    if args.command == "report":
        text = build_markdown_report(
            result_csv=args.result_csv,
            risk_json=args.risk_json,
            portfolio_json=args.portfolio_json,
        )
        if args.output_md:
            Path(args.output_md).write_text(text, encoding="utf-8")
            print(f"wrote {args.output_md}")
        else:
            print(text)
        return
