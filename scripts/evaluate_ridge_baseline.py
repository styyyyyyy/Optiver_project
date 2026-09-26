#!/usr/bin/env python3
"""Evaluate Ridge as a direct base model on 92 engineered features.

The model input is the audited ``full_clean`` feature set.  It predicts the
target directly; it is not a stacking model.  Imputation and standardization
are fitted inside each outer training fold, and complete ``time_id`` groups
remain together throughout five-fold OOF evaluation.

Example
-------
python scripts/evaluate_ridge_baseline.py \
  --features-path data/processed/features_phase2.parquet \
  --output-dir artifacts/ridge_full_clean_group5_seed2021
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optiver.feature_sets import get_feature_set  # noqa: E402
from optiver.metrics import (  # noqa: E402
    grouped_bootstrap_rmspe,
    rmspe,
    rmspe_weights,
)
from optiver.validation import build_group_folds  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.alpha < 0:
        raise ValueError("--alpha must be non-negative")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(args.features_path.expanduser().resolve())
    feature_names = get_feature_set("full_clean", frame.columns, strict=True)
    if len(feature_names) != 92:
        raise ValueError(
            f"Expected 92 full_clean features; resolved {len(feature_names)}"
        )

    folds = build_group_folds(
        frame[["time_id"]],
        n_splits=args.n_splits,
        random_state=args.seed,
    )
    target = frame["target"].to_numpy(dtype=np.float64)
    predictions = np.full(len(frame), np.nan, dtype=np.float64)
    fold_ids = np.full(len(frame), -1, dtype=np.int16)
    fold_rows: list[dict[str, float | int]] = []
    coefficient_rows: list[dict[str, float | int | str]] = []

    for fold in folds:
        train_index, valid_index = fold.train_idx, fold.valid_idx
        train_features = frame.iloc[train_index][feature_names].replace(
            [np.inf, -np.inf], np.nan
        )
        valid_features = frame.iloc[valid_index][feature_names].replace(
            [np.inf, -np.inf], np.nan
        )

        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(train_features).astype(np.float32)
        x_valid = imputer.transform(valid_features).astype(np.float32)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train).astype(np.float32)
        x_valid = scaler.transform(x_valid).astype(np.float32)

        model = Ridge(alpha=args.alpha)
        model.fit(
            x_train,
            target[train_index],
            sample_weight=rmspe_weights(target[train_index]),
        )
        fold_prediction = np.clip(model.predict(x_valid), 0.0, None)
        predictions[valid_index] = fold_prediction
        fold_ids[valid_index] = fold.fold_id
        fold_rows.append(
            {
                "fold": fold.fold_id,
                "n_rows": len(valid_index),
                "n_time_ids": len(fold.valid_groups),
                "rmspe": rmspe(target[valid_index], fold_prediction),
            }
        )
        coefficient_rows.extend(
            {
                "fold": fold.fold_id,
                "feature": feature_name,
                "standardized_coefficient": float(coefficient),
            }
            for feature_name, coefficient in zip(
                feature_names, model.coef_, strict=True
            )
        )

    if np.any(fold_ids < 0) or not np.all(np.isfinite(predictions)):
        raise RuntimeError("OOF predictions do not cover every row exactly once")

    interval = grouped_bootstrap_rmspe(
        target,
        predictions,
        frame["time_id"].to_numpy(),
        n_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    fold_metrics = pd.DataFrame(fold_rows)
    model_summary = pd.DataFrame(
        [
            {
                "model": "ridge_full_clean_92",
                "n_features": len(feature_names),
                "alpha": args.alpha,
                "pooled_oof_rmspe": interval.estimate,
                "ci_lower": interval.lower,
                "ci_upper": interval.upper,
                "confidence": interval.confidence,
                "bootstrap_resamples": interval.n_resamples,
                "mean_fold_rmspe": fold_metrics["rmspe"].mean(),
                "std_fold_rmspe": fold_metrics["rmspe"].std(ddof=0),
            }
        ]
    )

    oof = frame[["stock_id", "time_id", "target"]].copy()
    oof["fold"] = fold_ids
    oof["prediction"] = predictions
    oof.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(
        output_dir / "coefficients.csv", index=False
    )
    model_summary.to_csv(output_dir / "model_summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                **model_summary.iloc[0].to_dict(),
                "feature_set": "full_clean",
                "split_strategy": "group_kfold_by_time_id",
                "training_objective": "RMSPE-aligned weighted Ridge",
                "sample_weight": "normalized 1 / target^2",
                "prediction_clipping": "non-negative",
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(model_summary.to_string(index=False))
    print("\nFold metrics:")
    print(fold_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
