# Experiment protocol

## Objective

Predict the positive realized-volatility target and minimise root mean squared
percentage error (RMSPE):

```text
RMSPE = sqrt(mean(((target - prediction) / target) ** 2))
```

## Unit of validation

Every row belongs to a `(stock_id, time_id)` pair. All rows sharing a `time_id`
remain in the same fold. Random row-level splitting is prohibited because it
would mix the same market window between training and validation.

## Feature policy

- The clean registry contains observable book, trade and same-window market
  features.
- Cached columns created from full-target stock clustering (`stock_cluster`
  and names ending in `_<n>c1`) are rejected.
- Feature-based clustering is fitted only on the relevant training scope.
- Stock target statistics, when explicitly enabled, are cross-fitted in the
  training scope and never use validation labels.

## Candidate policy

- Baselines and feature ablations are evaluated under the same grouped fold map.
- Hyperparameter checks are development diagnostics, not independent
  confirmations.
- The final LightGBM component uses 256 leaves and 90 fixed rounds.
- The final MLP uses 44 curated inputs and a stock embedding.

## Strict ensemble policy

For each outer fold, split only its outer-training groups into an inner-fit and
calibration partition. Train calibration components on inner-fit, predict the
calibration partition, fit simplex weights there, and apply those weights to
the outer-fold OOF component predictions. Do not fit weights on post-hoc OOF
rows from other folds.

## Uncertainty

Resample complete `time_id` groups. Paired comparisons use the same resampled
groups for candidate and reference. Intervals are conditional on the complete
frozen development procedure and do not include model-selection uncertainty.

## Required run artifacts

Each formal run records:

- status and completion markers;
- effective parameters and environment versions;
- fold assignments;
- OOF predictions with keys, targets and fold IDs;
- aggregate and fold metrics;
- input, output and source-file SHA-256 hashes.
