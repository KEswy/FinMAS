# FinMAS v5 Progress

> 本文件用于持续跟踪 FinMAS v5 全链路重构。每次完成一个可验证子任务后更新。

## Snapshot

- Date: 2026-08-16
- Status: in progress
- Baseline: legacy 321-scenario walk-forward results
  - DeepSeek latest: 51.09% direction accuracy
  - Llama3.1 seed123: 51.71% direction accuracy
- Current goal: direction accuracy priority, full-link refactor, LLM as evidence/path extractor

## Done

- Confirmed current repository structure and 321-scenario baseline metrics.
- Confirmed 21 legacy tests pass.
- Confirmed existing git remote and SSH key availability.
- Created `finmas/`, `tests/`, and `legacy/` package directories.
- Added public contracts in `finmas/schemas.py`.
- Added `FeatureStore` and `TimeFirewall` for leak-safe feature construction.
- Added structured `LLMProvider` and `MechanismExtractor`.
- Added hybrid retrieval and dynamic in-memory event KG.
- Added time-series model pool with linear/MLP/momentum/XGBoost candidates.
- Added calibrated fusion, selective prediction, and risk engine.
- Added FastAPI and CLI entrypoints.
- Added initial v5 tests; full test suite currently 29 passing.
- Added deterministic walk-forward evaluation script and smoke-run verification.
- Added repository `README.md` and `DATA_LICENSE.md`.
- Updated `.gitignore` to include risk-center/report results while keeping `.env` excluded.

## In Progress

- End-to-end pipeline smoke test and API integration.
- Full 321-scenario multi-seed validation harness.
- GitHub branch preparation and secret audit.

## Next

- Wire the new pipeline into a repeatable evaluation script.
- Add paired bootstrap and selective-prediction reporting.
- Run baseline regression against legacy 321 results.
- Prepare `README`, `DATA_LICENSE`, and `FinMAS` branch.

## Blockers

- None.
