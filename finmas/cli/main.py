"""FinMAS v5 command-line interface."""

from __future__ import annotations

import argparse
import json

from ..pipeline import FinMASPipeline
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

