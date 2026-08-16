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
from ..sim.simulator import MarketSimulator
from ..sim.report import save_json, to_html


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

    sim = sub.add_parser("sim", help="continuous market simulation")
    sim_sub = sim.add_subparsers(dest="sim_command", required=True)
    sim_run = sim_sub.add_parser("run")
    sim_run.add_argument("--start", default="2024-01-02")
    sim_run.add_argument("--end", default="2024-03-01")
    sim_run.add_argument("--output-json", default="data/eval/sim_state.json")
    sim_run.add_argument("--output-html", default="data/eval/sim_report.html")
    sim_report = sim_sub.add_parser("report")
    sim_report.add_argument("--state-json", required=True)
    sim_report.add_argument("--output-html", required=True)
    sim_graph = sim_sub.add_parser("graph")
    sim_graph.add_argument("--state-json", required=True)

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

    if args.command == "sim":
        if args.sim_command == "run":
            import pandas as pd

            market = pd.read_csv("data/raw/hs300.csv")
            market["日期"] = pd.to_datetime(market["日期"])
            mask = (market["日期"] >= pd.Timestamp(args.start)) & (
                market["日期"] <= pd.Timestamp(args.end)
            )
            dates = market.loc[mask, "日期"].dt.strftime("%Y-%m-%d").tolist()
            sim_runner = MarketSimulator()
            state = sim_runner.run(dates)
            save_json(state, args.output_json)
            Path(args.output_html).write_text(to_html(state), encoding="utf-8")
            print(f"dates={len(dates)} output={args.output_json}")
            return
        if args.sim_command == "report":
            state = __import__("finmas.sim.schemas", fromlist=["SimulationState"]).SimulationState()
            # Load a minimal reconstruction for reporting purposes.
            import json as _json

            raw = _json.loads(Path(args.state_json).read_text(encoding="utf-8"))
            Path(args.output_html).write_text(
                "<pre>" + _json.dumps(raw, ensure_ascii=False, indent=2) + "</pre>",
                encoding="utf-8",
            )
            print(f"report={args.output_html}")
            return
        if args.sim_command == "graph":
            import json as _json

            raw = _json.loads(Path(args.state_json).read_text(encoding="utf-8"))
            print(_json.dumps(raw.get("trace", []), ensure_ascii=False, indent=2))
            return
