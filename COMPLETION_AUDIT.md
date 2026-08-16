# FinMAS v5 Completion Audit

This document audits the original implementation plan against the current repository state. It is intentionally honest: completed items are marked `DONE`, partial items `PARTIAL`, and missing items `OPEN`.

## Phase 0 — Repository, tracking, and publication

- `DONE` Created `PROJECT_PROGRESS.md`.
- `DONE` Created public remote `git@github.com:KEswy/FinMAS.git` and pushed branch `FinMAS`.
- `DONE` Added `README.md` and `DATA_LICENSE.md`.
- `DONE` Audited tracked files for real API keys; `.env` remains untracked.
- `DONE` Excluded `.env`, logs, and pid files while including full data/results.

## Phase 1 — Data and leak-free baseline

- `DONE` Baseline anchored to 321 expanded scenarios.
- `DONE` Primary horizon T+5; secondary T+1/T+20.
- `PARTIAL` Event expansion is present via `event_library_expanded.csv`, but no additional post-v4 events were added in this run.
- `DONE` Unified feature store includes market, valuation, sentiment, macro, and text features.
- `PARTIAL` Leakage tests cover time firewall and retrieval; not every feature path has an explicit truncation-invariance test.

## Phase 2 — RAG and dynamic event KG

- `DONE` Hybrid BM25 + TF-IDF retrieval implemented without random fallback.
- `DONE` Evidence chunks include `chunk_id`, `pub_time`, source, and metadata.
- `DONE` Dynamic in-memory event KG with time-decay edges implemented.
- `OPEN` Neo4j backend remains optional/not implemented; accepted by plan as in-memory first.

## Phase 3 — Structured LLM extraction

- `DONE` Unified LLM provider with DeepSeek/Ollama, JSON mode, retries, cache, and cost tracking.
- `DONE` `MechanismExtractor` emits factors, not final direction.
- `DONE` Text-factor fallback and multi-seed prompt perturbation implemented.
- `PARTIAL` Debate/RelDecomp are available in legacy code but not fully integrated into the new v5 evaluator as optional factors.

## Phase 4 — Time-series model pool

- `DONE` Momentum, linear, MLP, and XGBoost candidates implemented behind a common interface.
- `DONE` Temporal validation selection implemented.
- `PARTIAL` Full candidate-pool benchmark across all 321 events not yet executed as a separate experiment; the evaluator uses a lighter feature-based path.

## Phase 5 — Fusion, calibration, and selective prediction

- `DONE` Interpretable calibrated fusion implemented.
- `DONE` Temporal/isotonic/Platt calibration implemented.
- `DONE` Out-of-fold selective threshold calibration implemented.
- `DONE` Selective coverage/accuracy curve and per-event-type reporting implemented.
- `DONE` no-LLM, real-LLM, heuristic, ensemble, and walk-forward ensemble modes implemented.
- `DONE` Final recommended ensemble produced.

## Phase 6 — Risk module

- `DONE` Event-level VaR/ES/MDD estimation implemented.
- `DONE` T+1/T+5/T+20 risk backtests implemented and compared against legacy.
- `DONE` Portfolio and industry risk aggregation implemented for committed v5 rows.
- `PARTIAL` Stress tests and attribution are still primarily served by legacy `risk_center`; the new v5 module has basic attribution only.

## Phase 7 — CLI, API, and reports

- `DONE` New `finmas/` package created.
- `DONE` CLI `python -m finmas` implemented.
- `DONE` FastAPI endpoints implemented for health and prediction.
- `DONE` API integration tests added.
- `PARTIAL` Full report orchestration for new v5 results is split across scripts; no single `report` command yet.

## Test plan

- `DONE` 34 tests passing.
- `DONE` Legacy tests still pass in full suite.
- `DONE` Leakage tests added.
- `DONE` API integration tests added.
- `PARTIAL` 3-seed LLM run completed, but full multi-seed variance report across all components is still being assembled.
- `PARTIAL` Paired bootstrap implemented and run for key models, but not yet for every final sub-configuration.

## Acceptance gate

- `DONE` Full-sample accuracy of final ensemble is `56.39%`, above the legacy baseline `51.09%`.
- `DONE` Committed accuracy is `57.32%` with coverage `157/321`.
- `PARTIAL` Full-sample paired bootstrap p remains above `0.05`; improvement is directionally positive but not yet strictly significant.
- `DONE` T+1/T+5/T+20 VaR coverage is substantially better than legacy.

## Recommended remaining work

1. Run final ensemble portfolio/risk backtest with a cleaner out-of-fold scale calibration.
2. Add a single `finmas report` command that combines direction, risk, and portfolio outputs.
3. Add explicit truncation-invariance tests for valuation, sentiment, macro, and text features.
4. Integrate legacy Debate/RelDecomp into the new factor pipeline, or explicitly retire them.
5. Add a final multi-seed variance summary table for all final ensemble components.

