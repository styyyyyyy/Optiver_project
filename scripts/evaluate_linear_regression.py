#!/usr/bin/env python3
"""Evaluate the original Ridge linear-stacking baseline.

The script expects one parquet file containing ``target``, ``lgb_oof_pred``
and ``mlp_oof_pred``.  It reproduces the linear-regression section of the
original Phase 2C experiment and writes compact CSV results plus a figure.

This is an exploratory historical comparison.  Its shuffled row-level KFold
is not the strict nested grouped protocol used for the repository headline.

Example
-------
python scripts/evaluate_linear_regression.py \
  --predictions final_ensemble_predictions.parquet \
  --output-dir results/linear_regression_comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


REQUIRED_COLUMNS = ["target", "lgb_oof_pred", "mlp_oof_pred"]


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return RMSPE after clipping predictions to the non-negative domain."""

    prediction = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    target = np.asarray(y_true, dtype=float)
    if np.any(target <= 0):
        raise ValueError("RMSPE requires strictly positive targets")
    return float(np.sqrt(np.mean(np.square((target - prediction) / target))))


def evaluate(
    frame: pd.DataFrame,
    *,
    alpha: float = 0.01,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit shuffled-KFold Ridge stacking and return three result tables."""

    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    target = frame["target"].to_numpy(dtype=float)
    features = frame[["lgb_oof_pred", "mlp_oof_pred"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(features)):
        raise ValueError("Input columns contain NaN or infinite values")

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    stacked_prediction = np.empty_like(target)
    fold_rows: list[dict[str, float | int | str]] = []
    coefficient_rows: list[dict[str, float | int | str]] = []

    for fold, (train_index, valid_index) in enumerate(splitter.split(features)):
        model = Ridge(alpha=alpha)
        model.fit(features[train_index], target[train_index])
        stacked_prediction[valid_index] = np.clip(
            model.predict(features[valid_index]), 1e-8, None
        )
        fold_rows.append(
            {
                "fold": fold,
                "n_rows": len(valid_index),
                "rmspe": rmspe(target[valid_index], stacked_prediction[valid_index]),
            }
        )
        coefficient_rows.append(
            {
                "fit": f"fold_{fold}",
                "intercept": float(model.intercept_),
                "lightgbm_coefficient": float(model.coef_[0]),
                "mlp_coefficient": float(model.coef_[1]),
            }
        )

    full_model = Ridge(alpha=alpha).fit(features, target)
    coefficient_rows.append(
        {
            "fit": "all_rows_for_reference",
            "intercept": float(full_model.intercept_),
            "lightgbm_coefficient": float(full_model.coef_[0]),
            "mlp_coefficient": float(full_model.coef_[1]),
        }
    )

    summary = pd.DataFrame(
        [
            {"model": "LightGBM", "rmspe": rmspe(target, features[:, 0])},
            {"model": "MLP", "rmspe": rmspe(target, features[:, 1])},
            {
                "model": "Linear stacking (Ridge)",
                "rmspe": rmspe(target, stacked_prediction),
            },
        ]
    ).sort_values("rmspe", ignore_index=True)
    return summary, pd.DataFrame(fold_rows), pd.DataFrame(coefficient_rows)


def save_figure(summary: pd.DataFrame, output_path: Path) -> None:
    """Save a compact comparison plot."""

    figure, axis = plt.subplots(figsize=(8, 4.5))
    colors = ["#4C78A8", "#F58518", "#B8B8B8"]
    axis.barh(summary["model"], summary["rmspe"], color=colors)
    axis.invert_yaxis()
    axis.set_xlabel("RMSPE (lower is better)")
    axis.set_title("Historical linear-stacking comparison")
    for row_index, score in enumerate(summary["rmspe"]):
        axis.text(score, row_index, f"  {score:.6f}", va="center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.predictions.expanduser().resolve())
    summary, folds, coefficients = evaluate(
        frame,
        alpha=args.alpha,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    summary.to_csv(output_dir / "model_summary.csv", index=False)
    folds.to_csv(output_dir / "fold_metrics.csv", index=False)
    coefficients.to_csv(output_dir / "coefficients.csv", index=False)
    save_figure(summary, output_dir / "comparison.png")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
