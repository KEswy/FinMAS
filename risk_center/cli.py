"""Command-line entrypoints for risk-center reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .report import build_full_report
from .llm_commentary import generate_commentary


def _config_from_args(args: argparse.Namespace):
    config = load_config(args.config) if args.config else load_config()
    if args.result_csv:
        config.result_csv = args.result_csv
    return config


def cmd_report(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    report = build_full_report(config=config)
    output_dir = Path(config.output_dir)
    stem = Path(config.result_csv).stem
    print("risk_report=", output_dir / f"{stem}_risk_center.json")
    print("flat_csv=", output_dir / f"{stem}_risk_center_flat.csv")
    print("html=", output_dir / f"{stem}_risk_center_report.html")

    if args.commentary:
        try:
            commentary = generate_commentary(report)
            path = output_dir / f"{stem}_risk_commentary.md"
            path.write_text(commentary, encoding="utf-8")
            print("commentary=", path)
            print(commentary)
        except Exception as exc:
            print(f"commentary failed: {exc}")
    return 0


def cmd_stress(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    report = build_full_report(config=config)
    print(json.dumps(report["stress"], ensure_ascii=False, indent=2))
    return 0


def cmd_attribution(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    report = build_full_report(config=config)
    print(json.dumps(report["attribution"], ensure_ascii=False, indent=2)[:20000])
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import app

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Risk center analytics")
    sub = parser.add_subparsers(dest="command", required=True)

    report = sub.add_parser("report")
    report.add_argument("--config")
    report.add_argument("--result-csv")
    report.add_argument("--commentary", action="store_true")
    report.set_defaults(func=cmd_report)

    stress = sub.add_parser("stress")
    stress.add_argument("--config")
    stress.add_argument("--result-csv")
    stress.set_defaults(func=cmd_stress)

    attribution = sub.add_parser("attribution")
    attribution.add_argument("--config")
    attribution.add_argument("--result-csv")
    attribution.set_defaults(func=cmd_attribution)

    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
