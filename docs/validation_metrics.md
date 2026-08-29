# Validation results

## Degradation forecaster (measured on the `noisy_line` scenario)

| Lead time bucket (h) | n | Median abs. error (h) | IQR |
|---|---|---|---|
| 1 | 46 | 0.08 | 0.05-0.14 |
| 2 | 44 | 0.39 | 0.31-0.44 |
| 4 | 96 | 0.59 | 0.26-1.03 |
| 6 | 92 | 0.95 | 0.37-1.52 |
| 8 | 71 | 1.54 | 0.95-2.13 |

- **95% CI coverage:** 79.3% (target ~95%)
- **False forecast rate:** 0.03/shift (target < 0.5/shift)
- **Step-change refusal rate:** 84.2% over 19 episodes (target > 90%)

## Defect inference (measured on the `weld_drift_demo` scenario)

- Precision: 8.8% (3/34 flagged VINs had a real defect)
- Recall: 75.0% (3/4 real defects flagged)

## Detection lead time (measured on the `weld_drift_demo` scenario)

- First detection: 2026-01-01 09:55:25
- First real-world surfacing (EOL inspection): 2026-01-01 13:24:43
- **Lead time gained: 209 minutes**

Interactive plot: [forecast_error_vs_leadtime.html](forecast_error_vs_leadtime.html)
