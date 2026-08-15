# FinMAS

FinMAS is a firewall-safe, multi-agent financial analysis system for Chinese A-share macro-policy events and industry relative-return prediction. This repository contains the legacy v4 implementation, the v5 `finmas/` package, experiment data, and risk-center analytics.

## Structure

- `finmas/` — v5 refactor: structured LLM extraction, leak-safe feature store, hybrid retrieval, dynamic event KG, time-series model pool, calibrated fusion, risk engine, CLI, and API.
- `legacy/` — legacy v4 context pointer; the original v4 modules remain in the repository root for reproducible baselines.
- `agents/`, `rag/`, `models/`, `evaluation/`, `risk_center/` — legacy v4 implementation.
- `scripts/` — data preparation, experiment, and analysis scripts.
- `data/` — raw, processed, event, risk-center, and risk-report artifacts.

## Quick start

```powershell
python -m pip install -r requirements.txt
python -m finmas --help
python -m finmas predict --event-date 2024-07-22 --event-text "LPR下调" --event-type monetary_policy --industry-code 801780 --no-llm
python -m finmas serve
```

Run tests:

```powershell
pytest -q
```

## Security

`.env` is ignored and must never be committed. Do not add real `DEEPSEEK_API_KEY`, `LLM_API_KEY`, `ANTHROPIC_API_KEY`, or Bearer tokens to this public repository.

## Data provenance

See `DATA_LICENSE.md`. The included financial data and experiment artifacts are provided for research reproduction only.

