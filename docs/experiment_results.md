# Experiment results

## Headline

The current development headline is the strict nested calibration blend of a
LightGBM model and an MLP:

| Model | Pooled OOF RMSPE | Conditional grouped-bootstrap 95% CI |
|---|---:|---:|
| Strict nested blend | **0.2171231602** | [0.2129224897, 0.2224036512] |
| LightGBM | 0.2183344211 | [0.2138195436, 0.2241339619] |
| MLP | 0.2273455659 | [0.2237316094, 0.2320588293] |
| Ridge base model (92 engineered features) | 0.2358790999 | [0.2329645123, 0.2389871552] |

Against LightGBM, the paired RMSPE difference is
`-0.0012112609`, with conditional grouped-bootstrap 95% interval
`[-0.0019412613, -0.0005575088]`. Negative values favour the blend.

The machine-readable tables are in
[`results/final_nested_blend/`](../results/final_nested_blend/).

## Validation design

The primary evaluation uses five outer folds grouped by `time_id`. For each
outer fold:

1. The outer validation groups are removed.
2. The remaining 3,064 outer-training groups are split into 2,451 inner-fit
   groups and 613 calibration groups.
3. LightGBM and MLP calibration models train only on the inner-fit groups.
4. Their predictions on the disjoint calibration groups fit non-negative blend
   weights constrained to sum to one.
5. Those weights combine the already-frozen outer-OOF predictions for the
   outer validation fold.

No label from the scored outer fold enters component fitting, calibration
prediction or weight fitting for that fold. The five LightGBM weights were
approximately `0.726`, `0.608`, `0.712`, `0.616` and `0.844`.

## Component definitions

### Ridge base model

- direct prediction from the 92 audited `full_clean` engineered features
- no LightGBM or MLP predictions used as inputs
- fold-local median imputation and standardization
- Ridge `alpha=1.0` with RMSPE-aligned `1 / target^2` sample weights
- five-fold OOF grouped by `time_id`

### LightGBM

- 92 clean base features plus fold-fitted feature clustering and `stock_id`
- 256 leaves
- fixed 90 boosting rounds
- RMSPE-aligned sample weights
- five-fold grouped CV

### MLP

- 44 curated numerical features
- learned stock embedding
- hidden layers `256, 128, 64`
- per-outer-fold epoch selection inside the outer-training scope
- five-fold grouped CV

## Why older blend scores are not the headline

Earlier post-hoc blends produced RMSPE values around `0.216856` and `0.216236`.
They fit fold-specific weights using OOF predictions from other outer folds.
Those predictions came from base models whose training sets could include the
nominally held-out fold, creating an indirect dependency path. They remain
useful exploratory diagnostics but are not strict nested-CV estimates.

## Verification

- 64 unit and regression tests pass.
- All 428,932 OOF rows are covered exactly once.
- `(stock_id, time_id)` keys are unique and aligned across components.
- Recomputed fold weights match the stored weights to floating-point precision.
- Applying the stored weights reconstructs the final OOF predictions exactly.
- LightGBM and PyTorch calibration stages are process-isolated to avoid the
  observed macOS OpenMP runtime conflict.

## Limitations

1. There is no independent, unseen holdout result.
2. Model families, feature groups, 256 leaves and the decision to blend were
   informed by the same development dataset.
3. Bootstrap intervals condition on the frozen folds, trained models, selected
   candidates and fitted calibration weights. They do not rerun model selection
   or refit the complete study.
4. The calibration split contributes additional uncertainty that is not in the
   reported interval.
5. `time_id` has no reliable chronological meaning, so this is not a forward
   time-series test.
6. A full-data deployment-weight recipe has not been frozen.
7. The feature cache was audited but not regenerated from the roughly 200
   million raw order-book/trade rows during the final experiment cycle.

These constraints mean `0.217123` should be described as a **strict-nested
grouped-CV development estimate**, not as an independent test or production
performance claim.
