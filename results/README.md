# Committed results

This directory contains small, reviewable exports from the frozen local
experiment artifacts. Large OOF prediction files, fitted models and calibration
parquets remain under the ignored `artifacts/` directory.

## Files

- `experiment_results.csv`: machine-readable experiment ledger.
- `base_models/lightgbm_summary.json`: final LightGBM OOF summary.
- `base_models/mlp_summary.json`: final MLP OOF summary.
- `final_nested_blend/summary.json`: headline strict nested result.
- `final_nested_blend/model_summary.csv`: model scores and conditional intervals.
- `final_nested_blend/fold_metrics.csv`: per-fold model scores.
- `final_nested_blend/fold_weights.csv`: fold-local calibration weights and provenance hashes.
- `final_nested_blend/paired_comparisons.csv`: paired RMSPE differences and intervals.

The report in [`docs/experiment_results.md`](../docs/experiment_results.md)
explains which results are formal and which older results are exploratory.
