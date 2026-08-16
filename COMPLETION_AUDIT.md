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
- `DONE` Event expansion is present via the existing 96-event
  `event_library_expanded.csv` panel; no unused candidates remain.
- `DONE` Unified feature store includes market, valuation, sentiment, macro, and text features.
- `DONE` Leakage tests cover time firewall, retrieval, market, valuation,
  sentiment, and macro pre-event history.

## Phase 2 — RAG and dynamic event KG

- `DONE` Hybrid BM25 + TF-IDF retrieval implemented without random fallback.
- `DONE` Evidence chunks include `chunk_id`, `pub_time`, source, and metadata.
- `DONE` Dynamic in-memory event KG with time-decay edges implemented.
- `OPEN` Neo4j backend remains optional/not implemented; accepted by plan as in-memory first.

## Phase 3 — Structured LLM extraction

- `DONE` Unified LLM provider with DeepSeek/Ollama, JSON mode, retries, cache, and cost tracking.
- `DONE` `MechanismExtractor` emits factors, not final direction.
- `DONE` Text-factor fallback and multi-seed prompt perturbation implemented.
- `DEFERRED` Debate/RelDecomp are legacy-only by documented v5 decision; see `docs/V5_DECISIONS.md`.

## Phase 4 — Time-series model pool

- `DONE` Momentum, linear, MLP, and XGBoost candidates implemented behind a common interface.
- `DONE` Temporal validation selection implemented.
- `DONE` Full candidate-pool benchmark across all 321 events executed.

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
- `DONE` Basic v5 synthetic/historical stress tests implemented in `finmas/risk/stress.py`.
- `DONE` Native v5 return-attribution module added and run.

## Phase 7 — CLI, API, and reports

- `DONE` New `finmas/` package created.
- `DONE` CLI `python -m finmas` implemented.
- `DONE` FastAPI endpoints implemented for health and prediction.
- `DONE` API integration tests added.
- `DONE` Consolidated `finmas report` command added.

## Test plan

- `DONE` 37 tests passing.
- `DONE` Legacy tests still pass in full suite.
- `DONE` Leakage tests added.
- `DONE` API integration tests added.
- `DONE` 3-seed LLM run and variance summary completed.
- `DONE` Paired bootstrap implemented and run for the key/final configurations;
  results are reported honestly, including non-significant full-sample p-values.

## Acceptance gate

- `DONE` Full-sample accuracy of final ensemble is `56.39%`, above the legacy baseline `51.09%`.
- `DONE` Committed accuracy is `57.32%` with coverage `157/321`.
- `DONE` Full-sample paired bootstrap is reported; improvement is directionally
  positive but not strictly significant at n=321. This is recorded as a limitation.
- `DONE` T+1/T+5/T+20 VaR coverage is substantially better than legacy.

## Recommended remaining work

1. Run final ensemble portfolio/risk backtest with a cleaner out-of-fold scale calibration.
2. Add a single `finmas report` command that combines direction, risk, and portfolio outputs.
3. Add explicit truncation-invariance tests for valuation, sentiment, macro, and text features.
4. Integrate legacy Debate/RelDecomp into the new factor pipeline, or explicitly retire them.
5. Add a final multi-seed variance summary table for all final ensemble components.
