# FinMAS v5 Report

Result CSV: `data\processed\all_experiment_results_finmas_v5_final_ensemble.csv`
Rows: 321
Full accuracy: 56.39%  Committed accuracy: 57.32%  Coverage: 48.91%

## Selective curve
| Coverage | Accuracy | Min confidence |
|---:|---:|---:|
| 9.97% | 81.25% | 0.427 |
| 19.94% | 64.06% | 0.268 |
| 29.91% | 56.25% | 0.200 |
| 39.88% | 57.81% | 0.156 |
| 49.84% | 53.75% | 0.127 |

## Event-level risk
| Horizon | v5 breach | legacy breach | v5 Kupiec p |
|---:|---:|---:|---:|
| 1 | 4.36% | 11.84% | 0.592 |
| 5 | 7.48% | 13.71% | 0.057 |
| 20 | 5.92% | 18.38% | 0.462 |

## Portfolio risk
| Horizon | breach | Kupiec p | pinball |
|---:|---:|---:|---:|
| 1 | 3.18% | 0.265 | 0.00195 |
| 5 | 5.73% | 0.680 | 0.00280 |
| 20 | 4.46% | 0.751 | 0.00551 |