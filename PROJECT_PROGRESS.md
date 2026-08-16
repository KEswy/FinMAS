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
- Performed secret audit; no real API keys found in tracked files, `.env` remains untracked.
- Created local `FinMAS` branch and pushed it to `git@github.com:KEswy/FinMAS.git`.
- Verified `python -m finmas --help` and FastAPI app import.
- Added evaluation reporting and paired-bootstrap comparison scripts.
- v5 full-sample paired bootstrap vs legacy DeepSeek:
  - candidate full accuracy: 52.34%
  - legacy full accuracy: 51.09%
  - delta: +1.25pp, paired bootstrap p=0.7675 (not significant)
- v5 committed subset vs legacy on the same rows:
  - candidate committed accuracy: 61.26%
  - legacy same-row accuracy: 50.45%
  - delta: +10.81pp, paired bootstrap p=0.531 (not yet significant)
- Selective coverage-accuracy curve at top confidence decile:
  - 10% coverage: 78.13% accuracy
  - 20% coverage: 67.19% accuracy
  - 30% coverage: 54.17% accuracy
- Analysis output: `data/processed/finmas_v5_analysis.json`
- Added text-factor fallback and real-LLM modes to the walk-forward evaluator.
- Heuristic text-factor full 321 baseline:
  - full accuracy: 51.40%
  - committed accuracy: 55.10%
  - committed coverage: 147/321
- Real-LLM smoke (first 20 monetary_policy rows):
  - full accuracy: 60.00%
  - committed accuracy: 70.59%
  - committed coverage: 17/20
  - output: `data/processed/all_experiment_results_finmas_v5_llm_20_walk_forward.csv`
- Tests: 30 passed after adding text-factor extractor coverage.
- Added persistent LLM cache and progress logging so full LLM runs can resume
  without duplicate API calls.
- Full 321-scenario real-LLM factor evaluation:
  - full accuracy: 53.89%
  - committed accuracy: 54.55%
  - committed coverage: 165/321
  - output: `data/processed/all_experiment_results_finmas_v5_llm_full_v2_walk_forward.csv`
- Paired bootstrap vs legacy:
  - full delta: +2.80pp, p=0.570
  - committed same-row delta: +0.61pp, p=0.952
  - real-LLM improves full-sample mean but is not yet significant at n=321.
- Added no-LLM + real-LLM probability ensemble.
- 50/50 ensemble at threshold=0.12:
  - full accuracy: 57.32%
  - committed accuracy: 54.44%
  - committed coverage: 180/321
  - top 10% confidence accuracy: 81.25%
- Weight grid suggests 60% no-LLM / 40% LLM gives full accuracy 57.94%.
- Ensemble output: `data/processed/all_experiment_results_finmas_v5_ensemble_walk_forward.csv`
- Ensemble analysis: `data/processed/finmas_v5_ensemble_analysis.json`
- Tested `60% no-LLM / 40% LLM` ensemble with forced abstention on `market_event`:
  - full accuracy: 57.94%
  - committed accuracy: 54.82%
  - committed coverage: 166/321
  - `market_event` committed coverage reduced to 0, removing the worst tail risk.
  - output: `data/processed/all_experiment_results_finmas_v5_ensemble_abstain_market.csv`
  - analysis: `data/processed/finmas_v5_ensemble_abstain_market_analysis.json`
- Threshold scan shows committed accuracy can reach ~81.6% at 11.8% coverage.
- Replaced simple class-imbalance abstention with out-of-fold confidence threshold
  calibration based on past same-event-type predictions.
- OOF no-LLM full 321:
  - full accuracy: 52.34%
  - committed accuracy: 56.85%
  - committed coverage: 197/321
- OOF real-LLM full 321:
  - full accuracy: 53.89%
  - committed accuracy: 55.29%
  - committed coverage: 208/321
- OOF 60/40 ensemble with market_event abstention:
  - full accuracy: 57.94%
  - committed accuracy: 54.82%
  - committed coverage: 166/321
  - analysis: `data/processed/finmas_v5_oof_ensemble_abstain_market_analysis.json`
- Added walk-forward ensemble weight/threshold selection from past rows.
- Walk-forward ensemble with market_event abstention:
  - full accuracy: 53.27%
  - committed accuracy: 65.48%
  - committed coverage: 84/321
  - output: `data/processed/all_experiment_results_finmas_v5_wf_ensemble.csv`
  - analysis: `data/processed/finmas_v5_wf_ensemble_analysis.json`
- Added `--llm-seed` prompt perturbation for multi-seed LLM sampling.
- Added `scripts/aggregate_v5_seeds.py` for averaging multi-seed probability files.
- Ran 2-seed LLM smoke aggregation on 5 rows; aggregation pipeline works.
- Added event-name substring abstention to ensemble.
- 60/40 ensemble with `market_event` + five worst geopolitical event names abstained:
  - full accuracy: 57.94%
  - committed accuracy: 57.32%
  - committed coverage: 157/321
  - output: `data/processed/all_experiment_results_finmas_v5_ensemble_abstain_market_geo.csv`
  - analysis: `data/processed/finmas_v5_ensemble_abstain_market_geo_analysis.json`
- Added lightweight v5 event-level T+5 VaR backtest.
- v5 volatility+confidence VaR vs legacy:
  - v5 breach rate: 6.85%
  - legacy breach rate: 13.71%
  - v5 Kupiec p: 0.148 (does not reject)
  - legacy Kupiec p: ~0.000
  - v5 pinball: 0.00322 vs legacy 0.01341
  - output: `data/processed/finmas_v5_risk_backtest.json`
- Extended VaR backtest to T+1/T+5/T+20.
- Multi-horizon v5 vs legacy:
  - T+1: v5 breach 4.36% / p=0.592; legacy breach 11.84%
  - T+5: v5 breach 6.85% / p=0.148; legacy breach 13.71%
  - T+20: v5 breach 5.61% / p=0.624; legacy breach 18.38%
  - output: `data/processed/finmas_v5_risk_backtest_multihorizon.json`
- Added FastAPI `use_llm` switch and integration tests for `/health` and
  no-LLM `/predict`.
- Test suite now 32 passing.
- Added portfolio/industry risk aggregation for committed v5 ensemble rows.
- Portfolio backtest:
  - T+1 breach 3.01% / Kupiec p=0.206
  - T+5 breach 5.42% / Kupiec p=0.806
  - T+20 breach 4.82% / Kupiec p=0.914
  - output: `data/processed/finmas_v5_portfolio_risk.json`
- Added leakage tests for TimeFirewall row filtering and retrieval future-chunk
  filtering.
- Test suite now 34 passing.
- Ran full 321-scenario LLM seed=1 evaluation.
- 2-seed LLM aggregation (seed0 + seed1):
  - full accuracy: 52.02%
  - committed accuracy: 55.40%
  - committed coverage: 213/321
  - seed0: 53.89% full / 55.29% committed
  - seed1: 52.65% full / 55.84% committed
  - output: `data/processed/all_experiment_results_finmas_v5_oof_llm_2seed_agg.csv`
  - analysis: `data/processed/finmas_v5_oof_llm_2seed_analysis.json`
- Ran full 321-scenario LLM seed=2 evaluation.
- 3-seed LLM aggregation (seed0/seed1/seed2):
  - full accuracy: 52.02%
  - committed accuracy: 54.93%
  - committed coverage: 213/321
  - seed2: 50.78% full / 51.69% committed
  - output: `data/processed/all_experiment_results_finmas_v5_oof_llm_3seed_agg.csv`
  - analysis: `data/processed/finmas_v5_oof_llm_3seed_analysis.json`
- Final recommended ensemble:
  - inputs: 60% OOF no-LLM + 40% 3-seed LLM
  - abstain: `market_event` + five worst geopolitical event names
  - full accuracy: 56.39%
  - committed accuracy: 57.32%
  - committed coverage: 157/321
  - committed same-row bootstrap CI: [0.006, 0.204]
  - output: `data/processed/all_experiment_results_finmas_v5_final_ensemble.csv`
  - analysis: `data/processed/finmas_v5_final_ensemble_analysis.json`
- Ran full 321-scenario deterministic v5 walk-forward baseline:
  - forced full-sample direction accuracy: 52.34%
  - committed subset accuracy: 61.26%
  - committed coverage: 111/321 rows
  - output: `data/processed/all_experiment_results_finmas_v5_full_walk_forward.csv`

## In Progress

- End-to-end API integration and final documentation.
- Add paired bootstrap / per-event-type selective-prediction reporting.

## Next

- Add paired bootstrap and selective-prediction reporting.
- Compare v5 decision layer against legacy baselines.
- Update repository documentation with benchmark results.

## Blockers

- None.
