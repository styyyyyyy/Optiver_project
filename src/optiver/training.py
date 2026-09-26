"""Leakage-aware LightGBM cross-validation primitives.

The outer validation fold is never used to choose the number of boosting
iterations.  Each outer fold contains a group-preserving inner holdout used for
early stopping; the selected iteration count is then used to retrain on the
complete outer-training fold.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd

from .metrics import grouped_bootstrap_rmspe, lgb_rmspe, rmspe, rmspe_weights


IndexPair = tuple[np.ndarray, np.ndarray]
InnerSplitFactory = Callable[[np.ndarray, int], IndexPair]
FeatureProviderScope = Literal["inner_selection", "outer_refit"]
FoldFeatureProvider = Callable[
    [pd.DataFrame, pd.DataFrame, int, FeatureProviderScope],
    tuple[pd.DataFrame, pd.DataFrame, list[str]],
]


def _default_lgb_params() -> dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "None",
        "boosting_type": "gbdt",
        "num_leaves": 128,
        "learning_rate": 0.05,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 20,
        "lambda_l1": 0.5,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "num_threads": -1,
        "deterministic": True,
        "force_col_wise": True,
    }


@dataclass(frozen=True)
class LightGBMConfig:
    params: dict[str, Any] = field(default_factory=_default_lgb_params)
    max_boost_rounds: int = 3_000
    early_stopping_rounds: int = 150
    fixed_boost_rounds: int | None = None
    weight_clip_quantile: float | None = None
    seed: int = 42


@dataclass
class LightGBMCVResult:
    oof_predictions: np.ndarray
    fold_ids: np.ndarray
    fold_metrics: pd.DataFrame
    importance: pd.DataFrame
    summary: dict[str, float | int]
    feature_names: list[str]


def _validate_positions(indices: np.ndarray, n_rows: int, *, name: str) -> np.ndarray:
    positions = np.asarray(indices, dtype=np.int64)
    if positions.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(positions) == 0:
        raise ValueError(f"{name} must not be empty")
    if positions.min() < 0 or positions.max() >= n_rows:
        raise IndexError(f"{name} contains positions outside [0, {n_rows})")
    if len(np.unique(positions)) != len(positions):
        raise ValueError(f"{name} contains duplicate positions")
    return positions


def _sanitize_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric = result.select_dtypes(include=[np.number]).columns
    for column in numeric:
        # Assigning a float block through ``.loc`` into pandas 3 integer
        # columns raises instead of safely changing dtype. Column-wise
        # assignment makes the intended float32 conversion explicit.
        result[column] = (
            result[column]
            .replace([np.inf, -np.inf], np.nan)
            .astype(np.float32)
        )
    return result


def _fold_params(config: LightGBMConfig, fold: int) -> dict[str, Any]:
    seed = config.seed + fold
    params = dict(config.params)
    params.update(
        {
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
        }
    )
    return params


def _dataset(
    features: pd.DataFrame,
    targets: np.ndarray,
    *,
    config: LightGBMConfig,
    categorical_features: Sequence[str],
    reference: lgb.Dataset | None = None,
) -> lgb.Dataset:
    weights = rmspe_weights(
        targets,
        normalize=True,
        clip_quantile=config.weight_clip_quantile,
    )
    return lgb.Dataset(
        features,
        label=targets,
        weight=weights,
        categorical_feature=list(categorical_features),
        reference=reference,
        free_raw_data=False,
    )


def _select_boost_rounds(
    train_features: pd.DataFrame,
    train_targets: np.ndarray,
    valid_features: pd.DataFrame,
    valid_targets: np.ndarray,
    *,
    config: LightGBMConfig,
    params: dict[str, Any],
    categorical_features: Sequence[str],
) -> tuple[int, float]:
    train_set = _dataset(
        train_features,
        train_targets,
        config=config,
        categorical_features=categorical_features,
    )
    valid_set = _dataset(
        valid_features,
        valid_targets,
        config=config,
        categorical_features=categorical_features,
        reference=train_set,
    )
    model = lgb.train(
        params,
        train_set,
        num_boost_round=config.max_boost_rounds,
        valid_sets=[valid_set],
        valid_names=["inner_valid"],
        feval=lgb_rmspe,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=config.early_stopping_rounds,
                first_metric_only=True,
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iteration = max(int(model.best_iteration), 1)
    inner_prediction = model.predict(
        valid_features, num_iteration=best_iteration
    )
    inner_score = rmspe(valid_targets, inner_prediction)
    return best_iteration, inner_score


def _provide_features(
    feature_provider: FoldFeatureProvider,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    *,
    fold: int,
    scope: FeatureProviderScope,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Fit fold-dependent features without exposing validation labels.

    The provider receives the target on its fit partition only.  Its validation
    frame has the target column removed, which turns the intended fit scope into
    an enforced API boundary rather than a convention inside individual
    cluster/encoding implementations.
    """

    provider_valid = valid_frame.drop(columns=[target_col], errors="ignore").copy()
    x_train, x_valid, categorical_features = feature_provider(
        train_frame.copy(), provider_valid, fold, scope
    )
    if len(x_train) != len(train_frame) or len(x_valid) != len(valid_frame):
        raise ValueError(
            f"Fold {fold} ({scope}) feature provider changed the number of rows"
        )
    if list(x_train.columns) != list(x_valid.columns):
        raise ValueError(
            f"Fold {fold} ({scope}) train/validation feature schemas differ"
        )
    missing_categories = set(categorical_features).difference(x_train.columns)
    if missing_categories:
        raise ValueError(
            f"Fold {fold} ({scope}) has unknown categorical features: "
            f"{sorted(missing_categories)}"
        )
    return (
        _sanitize_features(x_train.reset_index(drop=True)),
        _sanitize_features(x_valid.reset_index(drop=True)),
        list(categorical_features),
    )


def run_lightgbm_cv(
    frame: pd.DataFrame,
    outer_folds: Sequence[IndexPair],
    *,
    feature_provider: FoldFeatureProvider,
    inner_split_factory: InnerSplitFactory,
    config: LightGBMConfig | None = None,
    target_col: str = "target",
    time_col: str = "time_id",
    model_dir: str | Path | None = None,
    allow_partial_oof: bool = False,
) -> LightGBMCVResult:
    """Run nested, group-preserving CV and return auditable OOF artifacts.

    ``allow_partial_oof`` is reserved for explicitly labelled calibration or
    pilot runs. Normal experiments require exactly-once validation coverage.
    """

    if config is None:
        config = LightGBMConfig()
    if target_col not in frame or time_col not in frame:
        raise ValueError(f"frame must contain {target_col!r} and {time_col!r}")
    targets_all = frame[target_col].to_numpy(dtype=np.float64)
    groups_all = frame[time_col].to_numpy()
    n_rows = len(frame)
    oof = np.full(n_rows, np.nan, dtype=np.float64)
    fold_ids = np.full(n_rows, -1, dtype=np.int16)
    fold_records: list[dict[str, float | int]] = []
    importance_records: list[pd.DataFrame] = []
    canonical_features: list[str] | None = None

    # Validate the complete outer-fold collection before fitting any feature
    # transformer or model.  This prevents an invalid late fold from leaving a
    # seemingly usable set of partial model artifacts behind.
    if not outer_folds:
        raise ValueError("outer_folds must contain at least one fold")
    validated_folds: list[IndexPair] = []
    validation_counts = np.zeros(n_rows, dtype=np.int16)
    all_positions = np.arange(n_rows, dtype=np.int64)
    for fold, (raw_train_idx, raw_valid_idx) in enumerate(outer_folds):
        train_idx = _validate_positions(raw_train_idx, n_rows, name="outer_train")
        valid_idx = _validate_positions(raw_valid_idx, n_rows, name="outer_valid")
        if np.intersect1d(train_idx, valid_idx).size:
            raise ValueError(f"Outer fold {fold} has overlapping train/validation rows")
        if not np.array_equal(
            np.sort(np.concatenate([train_idx, valid_idx])), all_positions
        ):
            raise ValueError(
                f"Outer fold {fold} train/validation rows are not complements"
            )
        train_groups = set(frame.iloc[train_idx][time_col].tolist())
        valid_groups = set(frame.iloc[valid_idx][time_col].tolist())
        overlap = train_groups.intersection(valid_groups)
        if overlap:
            preview = sorted((repr(value) for value in overlap))[:5]
            raise ValueError(
                f"Outer fold {fold} leaks {time_col} groups: {preview}"
            )
        validation_counts[valid_idx] += 1
        validated_folds.append((train_idx, valid_idx))

    invalid_coverage = (
        validation_counts > 1
        if allow_partial_oof
        else validation_counts != 1
    )
    bad_coverage = np.flatnonzero(invalid_coverage)
    if bad_coverage.size:
        coverage_contract = (
            "must cover every row exactly once"
            if not allow_partial_oof
            else "must not validate any row more than once"
        )
        raise ValueError(
            f"Outer validation folds {coverage_contract}; "
            f"bad positions: {bad_coverage[:10].tolist()}"
        )
    if allow_partial_oof and not np.any(validation_counts == 1):
        raise ValueError("Partial OOF run did not validate any rows")

    output_models = Path(model_dir) if model_dir is not None else None
    if output_models is not None:
        output_models.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, valid_idx) in enumerate(validated_folds):
        outer_train = frame.iloc[train_idx].copy()
        outer_valid = frame.iloc[valid_idx].copy()
        y_train = outer_train[target_col].to_numpy(dtype=np.float64)
        y_valid = outer_valid[target_col].to_numpy(dtype=np.float64)

        params = _fold_params(config, fold)
        if config.fixed_boost_rounds is not None:
            if config.fixed_boost_rounds < 1:
                raise ValueError("fixed_boost_rounds must be positive")
            best_iteration = int(config.fixed_boost_rounds)
            inner_score = float("nan")
        else:
            # Split the raw outer-training rows first.  Fold-fitted features
            # used for early stopping must be fitted on inner_train only; they
            # are deliberately rebuilt from the full outer_train afterwards.
            inner_train, inner_valid = inner_split_factory(
                outer_train[time_col].to_numpy(), config.seed + fold
            )
            inner_train = _validate_positions(
                inner_train, len(outer_train), name="inner_train"
            )
            inner_valid = _validate_positions(
                inner_valid, len(outer_train), name="inner_valid"
            )
            if np.intersect1d(inner_train, inner_valid).size:
                raise ValueError(f"Inner split for outer fold {fold} overlaps")
            if not np.array_equal(
                np.sort(np.concatenate([inner_train, inner_valid])),
                np.arange(len(outer_train), dtype=np.int64),
            ):
                raise ValueError(
                    f"Inner split for outer fold {fold} is not a full partition"
                )
            train_groups = set(outer_train.iloc[inner_train][time_col])
            valid_groups = set(outer_train.iloc[inner_valid][time_col])
            if train_groups.intersection(valid_groups):
                raise ValueError(
                    f"Inner split for outer fold {fold} leaks time_id groups"
                )
            inner_train_frame = outer_train.iloc[inner_train].copy()
            inner_valid_frame = outer_train.iloc[inner_valid].copy()
            (
                x_inner_train,
                x_inner_valid,
                inner_categorical_features,
            ) = _provide_features(
                feature_provider,
                inner_train_frame,
                inner_valid_frame,
                fold=fold,
                scope="inner_selection",
                target_col=target_col,
            )
            best_iteration, inner_score = _select_boost_rounds(
                x_inner_train,
                inner_train_frame[target_col].to_numpy(dtype=np.float64),
                x_inner_valid,
                inner_valid_frame[target_col].to_numpy(dtype=np.float64),
                config=config,
                params=params,
                categorical_features=inner_categorical_features,
            )

        x_train, x_valid, categorical_features = _provide_features(
            feature_provider,
            outer_train,
            outer_valid,
            fold=fold,
            scope="outer_refit",
            target_col=target_col,
        )
        if config.fixed_boost_rounds is None:
            if list(x_inner_train.columns) != list(x_train.columns):
                raise ValueError(
                    f"Fold {fold} feature schema changed between inner selection "
                    "and outer refit"
                )
            if inner_categorical_features != categorical_features:
                raise ValueError(
                    f"Fold {fold} categorical schema changed between inner "
                    "selection and outer refit"
                )
        if canonical_features is None:
            canonical_features = list(x_train.columns)
        elif canonical_features != list(x_train.columns):
            raise ValueError("Feature schema changed across outer folds")

        full_train_set = _dataset(
            x_train,
            y_train,
            config=config,
            categorical_features=categorical_features,
        )
        model = lgb.train(
            params,
            full_train_set,
            num_boost_round=best_iteration,
            callbacks=[lgb.log_evaluation(period=0)],
        )
        predictions = np.clip(
            model.predict(x_valid, num_iteration=best_iteration), 0.0, None
        )
        outer_score = rmspe(y_valid, predictions)
        oof[valid_idx] = predictions
        fold_ids[valid_idx] = fold

        fold_records.append(
            {
                "fold": fold,
                "train_rows": len(train_idx),
                "valid_rows": len(valid_idx),
                "train_time_ids": outer_train[time_col].nunique(),
                "valid_time_ids": outer_valid[time_col].nunique(),
                "best_iteration": best_iteration,
                "inner_rmspe": inner_score,
                "outer_rmspe": outer_score,
            }
        )
        gain = model.feature_importance(importance_type="gain").astype(float)
        gain_sum = gain.sum()
        normalized_gain = gain / gain_sum if gain_sum > 0 else gain
        importance_records.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "feature": x_train.columns,
                    "normalized_gain": normalized_gain,
                }
            )
        )
        if output_models is not None:
            model.save_model(str(output_models / f"fold_{fold}.txt"))

    oof_mask = fold_ids >= 0
    if (not allow_partial_oof and np.any(~oof_mask)) or np.any(
        ~np.isfinite(oof[oof_mask])
    ):
        missing = int(np.count_nonzero(fold_ids < 0))
        raise ValueError(f"Outer folds did not produce exactly one OOF prediction per row: {missing}")

    fold_metrics = pd.DataFrame(fold_records)
    raw_importance = pd.concat(importance_records, ignore_index=True)
    importance = (
        raw_importance.groupby("feature", as_index=False)["normalized_gain"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
        .rename(columns={"mean": "normalized_gain_mean", "std": "normalized_gain_std"})
    )
    interval = grouped_bootstrap_rmspe(
        targets_all[oof_mask],
        oof[oof_mask],
        groups_all[oof_mask],
        n_resamples=1_000,
        seed=config.seed,
    )
    summary: dict[str, float | int] = {
        "n_rows": n_rows,
        "n_oof_rows": int(oof_mask.sum()),
        "oof_fraction": float(oof_mask.mean()),
        "partial_run": bool(allow_partial_oof),
        "n_features": len(canonical_features or []),
        "n_folds": len(validated_folds),
        "mean_fold_rmspe": float(fold_metrics["outer_rmspe"].mean()),
        "std_fold_rmspe": float(fold_metrics["outer_rmspe"].std(ddof=0)),
        "pooled_oof_rmspe": interval.estimate,
        "group_bootstrap_ci_lower": interval.lower,
        "group_bootstrap_ci_upper": interval.upper,
    }
    return LightGBMCVResult(
        oof_predictions=oof,
        fold_ids=fold_ids,
        fold_metrics=fold_metrics,
        importance=importance,
        summary=summary,
        feature_names=canonical_features or [],
    )


__all__ = [
    "FeatureProviderScope",
    "FoldFeatureProvider",
    "IndexPair",
    "InnerSplitFactory",
    "LightGBMConfig",
    "LightGBMCVResult",
    "run_lightgbm_cv",
]
