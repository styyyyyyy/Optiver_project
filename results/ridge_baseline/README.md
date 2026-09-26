# Ridge base model

This experiment uses Ridge as a direct base model:

| Item | Definition |
|---|---|
| Input `X` | 92 audited `full_clean` engineered features |
| Target `y` | realized-volatility target |
| Model role | direct linear prediction, not stacking |
| Validation | five-fold OOF grouped by `time_id` |
| Objective alignment | training weights proportional to `1 / target^2` |
| Ridge alpha | `1.0` |

The pooled OOF RMSPE is `0.235879`, with a conditional grouped-bootstrap 95%
interval of `[0.232965, 0.238987]`. Median imputation and standardization are
fitted separately inside each training fold.

The code is [`scripts/evaluate_ridge_baseline.py`](../../scripts/evaluate_ridge_baseline.py).
This directory commits only the small reviewable tables; row-level OOF
predictions remain under the ignored `artifacts/` directory.
