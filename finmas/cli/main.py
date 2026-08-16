"""FinMAS v5 command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..pipeline import FinMASPipeline
from ..graph import FinMASGraph
from ..quality import build_rag_ground_truth, evaluate_rag
from ..quality.reports import generate_all_quality_reports
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

    graph = sub.add_parser("graph", help="LangGraph orchestration")
    graph_sub = graph.add_subparsers(dest="graph_command", required=True)
    graph_run = graph_sub.add_parser("run")
    graph_run.add_argument("--event-date", required=True)
    graph_run.add_argument("--event-text", required=True)
    graph_run.add_argument("--event-type", required=True)
    graph_run.add_argument("--industry-code", required=True)
    graph_run.add_argument("--no-llm", action="store_true")
    graph_run.add_argument("--debate-rounds", type=int, default=2)

    quality = sub.add_parser("quality", help="module quality reports")
    quality_sub = quality.add_subparsers(dest="quality_command", required=True)
    quality_rag = quality_sub.add_parser("rag")
    quality_rag.add_argument("--build", action="store_true")
    quality_rag.add_argument("--output-json", default="data/eval/rag_report.json")
    quality_all = quality_sub.add_parser("all")
    quality_all.add_argument(
        "--result-csv",
        default="data/processed/all_experiment_results_finmas_v5_final_ensemble.csv",
    )
    quality_all.add_argument(
        "--risk-json",
        default="data/processed/finmas_v5_final_risk_backtest.json",
    )

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

    if args.command == "graph":
        if args.graph_command != "run":
            return
        event = EventInput(
            event_date=args.event_date,
            event_text=args.event_text,
            event_type=args.event_type,
            industry_code=args.industry_code,
        )
        pipeline = FinMASPipeline(use_llm=not args.no_llm)
        graph_runner = FinMASGraph(
            pipeline=pipeline,
            debate_rounds=args.debate_rounds,
        )
        state = graph_runner.run(event)
        print(json.dumps(state.to_dict(), ensure_ascii=False, indent=2, default=str))
        return

    if args.command == "quality":
        if args.quality_command == "rag":
            if args.build:
                samples = build_rag_ground_truth()
                print(f"built {len(samples)} RAG samples")
            from ..rag.retrieval import build_retriever

            report = evaluate_rag(build_retriever())
            text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
            Path(args.output_json).write_text(text, encoding="utf-8")
            print(text)
            return
        if args.quality_command == "all":
            reports = generate_all_quality_reports(
                result_csv=args.result_csv,
                risk_json=args.risk_json,
            )
            print(json.dumps(reports, ensure_ascii=False, indent=2, default=str))
            return
