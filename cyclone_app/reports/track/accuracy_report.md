# Accuracy report (track model)

Data: 421 cyclones / 7220 best-track fixes (IMD, 1982-2026). Every number below is measured on
cyclones the model was **not trained on**. Baselines: track = constant motion of the last 6 h; wind / pressure = persistence.

## 5-fold cross-validation (whole cyclones held out, all years)
| Lead (h) | Track err (km) | Baseline (km) | Within 100 km | Wind MAE (kt) | Persist. (kt) | Within ±10 kt | Class correct | Pressure MAE (hPa) | Persist. (hPa) | Within ±5 hPa | ΔP MAE (hPa) | Still-active acc. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| +6 | 51 | 59 | 91% | 3.0 | 3.3 | 95% | 77% | 1.8 | 2.0 | 92% | 1.6 | 94% |
| +12 | 88 | 110 | 67% | 5.1 | 6.3 | 86% | 63% | 3.0 | 3.8 | 84% | 2.7 | 89% |
| +18 | 126 | 162 | 47% | 6.8 | 9.1 | 78% | 54% | 4.1 | 5.4 | 76% | 3.8 | 86% |
| +24 | 167 | 219 | 29% | 8.7 | 11.6 | 70% | 48% | 5.3 | 6.9 | 67% | 4.9 | 83% |
| +30 | 212 | 276 | 19% | 11.0 | 14.2 | 61% | 39% | 6.6 | 8.4 | 61% | 6.2 | 80% |
| +36 | 251 | 335 | 14% | 12.3 | 16.0 | 57% | 38% | 7.4 | 9.6 | 57% | 7.1 | 78% |
| +42 | 293 | 392 | 10% | 13.9 | 17.9 | 54% | 35% | 8.4 | 10.8 | 52% | 8.3 | 77% |
| +48 | 338 | 456 | 7% | 15.0 | 19.2 | 51% | 32% | 9.3 | 11.7 | 49% | 9.0 | 76% |

## Forward hold-out: trained 1982-2022, tested on 2023-2026 (38 cyclones)
| Lead (h) | Track err (km) | Baseline (km) | Within 100 km | Wind MAE (kt) | Persist. (kt) | Within ±10 kt | Class correct | Pressure MAE (hPa) | Persist. (hPa) | Within ±5 hPa | ΔP MAE (hPa) | Still-active acc. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| +6 | 35 | 37 | 97% | 2.5 | 2.9 | 96% | 80% | 1.2 | 1.5 | 96% | 1.2 | 94% |
| +12 | 68 | 75 | 80% | 4.3 | 5.6 | 89% | 67% | 2.1 | 2.9 | 90% | 1.9 | 91% |
| +18 | 100 | 115 | 61% | 5.6 | 8.0 | 84% | 58% | 3.2 | 4.2 | 83% | 2.7 | 88% |
| +24 | 136 | 158 | 40% | 7.6 | 10.1 | 73% | 49% | 4.2 | 5.3 | 75% | 4.0 | 87% |
| +30 | 175 | 205 | 28% | 9.7 | 11.6 | 66% | 41% | 5.3 | 6.1 | 71% | 4.9 | 78% |
| +36 | 208 | 251 | 20% | 10.6 | 12.8 | 61% | 39% | 6.4 | 6.8 | 63% | 5.9 | 74% |
| +42 | 238 | 298 | 17% | 12.2 | 13.9 | 59% | 32% | 6.8 | 7.5 | 64% | 7.5 | 74% |
| +48 | 278 | 345 | 13% | 13.7 | 14.8 | 53% | 32% | 7.7 | 7.9 | 60% | 7.4 | 72% |

## Land-crossing accuracy (cross-validation)
| | Forecasts | Landfalls detected | False alarms | Landfall-time error | Within ±6 h | Position error |
|---|---|---|---|---|---|---|
| Model | 4713 | 71% | 20% | 5.3 h | 69% | 92 km |
| Baseline | 4713 | 67% | 30% | 5.7 h | 66% | 102 km |

## Land-crossing accuracy (hold-out 2023-2026)
| | Forecasts | Landfalls detected | False alarms | Landfall-time error | Within ±6 h | Position error |
|---|---|---|---|---|---|---|
| Model | 526 | 81% | 28% | 5.1 h | 68% | 69 km |
| Baseline | 526 | 79% | 24% | 5.4 h | 67% | 78 km |

*Landfall-time / position error is for forecasts where both the forecast and the observed track cross the coast.*
