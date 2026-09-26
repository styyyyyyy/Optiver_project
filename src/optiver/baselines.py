"""Leakage-safe analytical baselines evaluated on fixed outer folds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import (
    fit_rmspe_scale,
    grouped_bootstrap_rmspe,
    optimal_rmspe_constant,
    rmspe,
)
from .training import IndexPair


@dataclass
class BaselineCVResult:
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    summary: pd.DataFrame


def _stock_scales(
    train: pd.DataFrame,
    *,
    stock_col: str,
    target_col: str,
    signal_col: str,
) -> pd.Series:
    ratio = train[signal_col].to_numpy(dtype=float) / train[target_col].to_numpy(
        dtype=float
    )
    work = pd.DataFrame(
        {
            stock_col: train[stock_col].to_numpy(),
            "numerator": ratio,
            "denominator": np.square(ratio),
        }
    )
    totals = work.groupby(stock_col, sort=False)[["numerator", "denominator"]].sum()
    return totals["numerator"] / totals["denominator"]


def _stock_optimal_constants(
    train: pd.DataFrame,
    *,
    stock_col: str,
    target_col: str,
) -> pd.Series:
    """Fit one RMSPE-optimal constant per stock on the training partition."""

    targets = train[target_col].to_numpy(dtype=float)
    work = pd.DataFrame(
        {
            stock_col: train[stock_col].to_numpy(),
            "inverse_target": np.reciprocal(targets),
            "inverse_target_squared": np.reciprocal(np.square(targets)),
        }
    )
    totals = work.groupby(stock_col, sort=False)[
        ["inverse_target", "inverse_target_squared"]
    ].sum()
    return totals["inverse_target"] / totals["inverse_target_squared"]


def evaluate_baselines(
    frame: pd.DataFrame,
    outer_folds: Sequence[IndexPair],
    *,
    target_col: str = "target",
    signal_col: str = "rv_pred",
    stock_col: str = "stock_id",
    time_col: str = "time_id",
    seed: int = 42,
) -> BaselineCVResult:
    required = {target_col, signal_col, stock_col, time_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing baseline columns: {missing}")

    n_rows = len(frame)
    methods = [
        "train_median",
        "rmspe_optimal_constant",
        "stock_optimal_constant",
        "raw_persistence",
        "global_calibrated_persistence",
        "stock_calibrated_persistence",
    ]
    predictions = {method: np.full(n_rows, np.nan, dtype=float) for method in methods}
    fold_ids = np.full(n_rows, -1, dtype=np.int16)
    fold_records: list[dict[str, float | int | str]] = []

    for fold, (train_idx, valid_idx) in enumerate(outer_folds):
        train = frame.iloc[np.asarray(train_idx, dtype=int)]
        valid = frame.iloc[np.asarray(valid_idx, dtype=int)]
        y_train = train[target_col].to_numpy(dtype=float)
        y_valid = valid[target_col].to_numpy(dtype=float)
        x_train = train[signal_col].to_numpy(dtype=float)
        x_valid = valid[signal_col].to_numpy(dtype=float)

        global_scale = fit_rmspe_scale(y_train, x_train)
        global_constant = optimal_rmspe_constant(y_train)
        per_stock_constant = _stock_optimal_constants(
            train,
            stock_col=stock_col,
            target_col=target_col,
        )
        validation_constants = valid[stock_col].map(per_stock_constant).fillna(
            global_constant
        )
        per_stock_scale = _stock_scales(
            train,
            stock_col=stock_col,
            target_col=target_col,
            signal_col=signal_col,
        )
        validation_scales = valid[stock_col].map(per_stock_scale).fillna(global_scale)

        fold_predictions = {
            "train_median": np.full(len(valid), np.median(y_train)),
            "rmspe_optimal_constant": np.full(
                len(valid), global_constant
            ),
            "stock_optimal_constant": validation_constants.to_numpy(),
            "raw_persistence": x_valid,
            "global_calibrated_persistence": global_scale * x_valid,
            "stock_calibrated_persistence": validation_scales.to_numpy() * x_valid,
        }
        for method, values in fold_predictions.items():
            values = np.clip(np.asarray(values, dtype=float), 0.0, None)
            predictions[method][valid_idx] = values
            fold_records.append(
                {
                    "fold": fold,
                    "method": method,
                    "valid_rows": len(valid),
                    "valid_time_ids": valid[time_col].nunique(),
                    "rmspe": rmspe(y_valid, values),
                }
            )
        fold_ids[valid_idx] = fold

    if np.any(fold_ids < 0):
        raise ValueError("Outer folds did not cover every row exactly once")
    prediction_frame = frame[[stock_col, time_col, target_col]].copy()
    prediction_frame["fold"] = fold_ids
    for method, values in predictions.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Baseline {method} has missing OOF predictions")
        prediction_frame[method] = values

    summaries: list[dict[str, float | int | str]] = []
    for method in methods:
        interval = grouped_bootstrap_rmspe(
            prediction_frame[target_col].to_numpy(),
            prediction_frame[method].to_numpy(),
            prediction_frame[time_col].to_numpy(),
            n_resamples=1_000,
            seed=seed,
        )
        summaries.append(
            {
                "method": method,
                "pooled_oof_rmspe": interval.estimate,
                "ci_lower": interval.lower,
                "ci_upper": interval.upper,
                "n_rows": n_rows,
                "n_folds": len(outer_folds),
            }
        )
    return BaselineCVResult(
        predictions=prediction_frame,
        fold_metrics=pd.DataFrame(fold_records),
        summary=pd.DataFrame(summaries).sort_values("pooled_oof_rmspe"),
    )


__all__ = ["BaselineCVResult", "evaluate_baselines"]
