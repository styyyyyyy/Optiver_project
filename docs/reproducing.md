# Reproducing the final result

## 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[mlp,plots]'
```

The frozen run used Python 3.12, LightGBM 4.x and CPU PyTorch. Exact package
versions and platform details are written into each generated run directory.

## 2. Data

Follow [`data/README.md`](../data/README.md), then verify the cache:

```bash
python scripts/verify_feature_cache.py \
  data/processed/features_phase2.parquet
```

The canonical file contains 428,932 rows, 112 stocks and 3,830 `time_id`
groups. Its SHA-256 is recorded in the manifest.

## 3. Tests

```bash
python -m unittest discover -s tests
```

## 4. Outer LightGBM OOF

```bash
python scripts/run_experiments.py lgbm \
  --features-path data/processed/features_phase2.parquet \
  --output-root artifacts \
  --run-name final_candidate_c1i1_leaves256_group5_seed2021 \
  --feature-set full_clean \
  --cluster-mode feature \
  --include-stock-id \
  --fixed-rounds 90 \
  --num-leaves 256 \
  --num-threads 4 \
  --seed 2021
```

## 5. Outer MLP OOF

```bash
python scripts/run_mlp_experiment.py \
  --features-path data/processed/features_phase2.parquet \
  --output-root artifacts \
  --run-name mlp_curated_nested_group5_seed2021 \
  --feature-set curated_mlp \
  --batch-size 8192 \
  --max-epochs 80 \
  --patience 10 \
  --device cpu \
  --seed 2021
```

Do not run LightGBM and PyTorch numerical work in the same Python interpreter
on macOS. The provided scripts are intentionally separate processes.

## 6. Fold-local calibration bundles

Run the following for each `FOLD` in `0 1 2 3 4`:

```bash
FOLD=0

python scripts/run_nested_calibration_mlp_fold.py \
  --features-path data/processed/features_phase2.parquet \
  --lightgbm-oof artifacts/final_candidate_c1i1_leaves256_group5_seed2021 \
  --mlp-oof artifacts/mlp_curated_nested_group5_seed2021 \
  --outer-fold "$FOLD" \
  --output-dir "artifacts/nested_calibration_mlp_seed2021_fold${FOLD}" \
  --device cpu \
  --seed 2021

python scripts/run_nested_calibration_fold.py \
  --features-path data/processed/features_phase2.parquet \
  --lightgbm-oof artifacts/final_candidate_c1i1_leaves256_group5_seed2021 \
  --mlp-oof artifacts/mlp_curated_nested_group5_seed2021 \
  --mlp-calibration-run "artifacts/nested_calibration_mlp_seed2021_fold${FOLD}" \
  --outer-fold "$FOLD" \
  --output-dir "artifacts/nested_calibration_seed2021_fold${FOLD}" \
  --num-threads 2 \
  --seed 2021
```

Each output directory must be new or empty. This prevents accidental mixing of
artifacts from different runs.

## 7. Strict nested evaluation

```bash
python scripts/evaluate_nested_blend.py \
  --model lightgbm=artifacts/final_candidate_c1i1_leaves256_group5_seed2021 \
  --model mlp=artifacts/mlp_curated_nested_group5_seed2021 \
  --calibration-bundle artifacts/nested_calibration_seed2021_fold0 \
  --calibration-bundle artifacts/nested_calibration_seed2021_fold1 \
  --calibration-bundle artifacts/nested_calibration_seed2021_fold2 \
  --calibration-bundle artifacts/nested_calibration_seed2021_fold3 \
  --calibration-bundle artifacts/nested_calibration_seed2021_fold4 \
  --output-dir artifacts/nested_blend_c1i1_lgb256_mlp_seed2021 \
  --bootstrap-resamples 5000 \
  --seed 2021
```

Expected pooled OOF RMSPE: `0.2171231602226673`.

## 8. Rebuild the public figure

```bash
python scripts/plot_results.py
```

## Scope of reproducibility

These commands reproduce the frozen candidate evaluation from the canonical
feature cache. They do not recreate the cache from all raw order-book/trade
rows, rerun the entire model-selection history, create independent test data,
or define production deployment weights.
