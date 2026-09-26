"""Post-hoc blending of out-of-fold predictions.

This module retains a useful descriptive OOF analysis, but it is *not* a
strictly nested, unbiased ensemble evaluation.  For outer fold ``k``, the
post-hoc method fits weights on OOF rows from the other folds.  Those rows were
predicted by base models whose training sets normally included fold ``k``.
Their predictions can therefore depend indirectly on data in the nominally
held-out fold.  Use :mod:`optiver.nested_blending` for strict evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import (
    grouped_bootstrap_rmspe,
    grouped_bootstrap_rmspe_difference,
    rmspe,
    validate_positive_targets,
)


KEY_COLUMNS = ("stock_id", "time_id")
REQUIRED_COLUMNS = (*KEY_COLUMNS, "target", "fold", "prediction")


@dataclass(frozen=True)
class AlignedOOF:
    """Strictly aligned component OOF predictions."""

    keys: pd.DataFrame
    target: np.ndarray
    fold: np.ndarray
    predictions: np.ndarray
    model_names: tuple[str, ...]


@dataclass(frozen=True)
class PosthocOOFBlendResult:
    """Artifacts from descriptive post-hoc OOF blending."""

    oof_predictions: np.ndarray
    fold_weights: pd.DataFrame
    deployment_weights: pd.DataFrame
    fold_metrics: pd.DataFrame
    model_summary: pd.DataFrame
    paired_comparisons: pd.DataFrame
    summary: dict[str, object]


def _validate_model_name(name: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Every component model must have a non-empty name")
    if name == "blend":
        raise ValueError("'blend' is reserved for the cross-fitted ensemble")


def _validated_frame(frame: pd.DataFrame, *, model_name: str) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(
            f"{model_name} OOF frame is missing columns: {sorted(missing)}"
        )
    result = frame.loc[:, REQUIRED_COLUMNS].copy()
    if result[list(KEY_COLUMNS)].isna().any(axis=None):
        raise ValueError(f"{model_name} contains null alignment keys")
    if result.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{model_name} contains duplicate stock_id/time_id keys")

    target = result["target"].to_numpy(dtype=np.float64)
    validate_positive_targets(target)
    prediction = result["prediction"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(prediction)):
        raise ValueError(f"{model_name} contains non-finite OOF predictions")

    raw_fold = result["fold"].to_numpy()
    try:
        fold_float = raw_fold.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{model_name} fold values must be integers") from exc
    if not np.all(np.isfinite(fold_float)) or not np.all(fold_float == np.floor(fold_float)):
        raise ValueError(f"{model_name} fold values must be finite integers")
    fold = fold_float.astype(np.int64)
    if np.any(fold < 0):
        raise ValueError(
            f"{model_name} is a partial OOF artifact; every row must have fold >= 0"
        )
    result["target"] = target
    result["fold"] = fold
    result["prediction"] = prediction
    return result


def align_oof_frames(frames: Mapping[str, pd.DataFrame]) -> AlignedOOF:
    """Align component OOF frames and reject every schema/content mismatch.

    Alignment is by ``(stock_id, time_id)`` rather than input row order.  The
    target and outer-fold assignment must match exactly after alignment.
    """

    if len(frames) < 2:
        raise ValueError("Cross-fitted blending requires at least two models")
    names = tuple(frames)
    if len(set(names)) != len(names):
        raise ValueError("Component model names must be unique")
    for name in names:
        _validate_model_name(name)

    validated = {
        name: _validated_frame(frame, model_name=name)
        for name, frame in frames.items()
    }
    reference_name = names[0]
    reference = validated[reference_name]
    reference_index = pd.MultiIndex.from_frame(reference[list(KEY_COLUMNS)])
    predictions: list[np.ndarray] = []

    for name in names:
        candidate = validated[name]
        candidate_index = pd.MultiIndex.from_frame(candidate[list(KEY_COLUMNS)])
        missing = reference_index.difference(candidate_index)
        extra = candidate_index.difference(reference_index)
        if len(missing) or len(extra) or len(candidate) != len(reference):
            raise ValueError(
                f"Key coverage differs for {name}: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
        if name == reference_name:
            aligned = candidate
        else:
            aligned = candidate.set_index(list(KEY_COLUMNS)).loc[reference_index].reset_index()

        candidate_target = aligned["target"].to_numpy(dtype=np.float64)
        if not np.array_equal(candidate_target, reference["target"].to_numpy(dtype=np.float64)):
            raise ValueError(f"Targets differ for {name}")
        candidate_fold = aligned["fold"].to_numpy(dtype=np.int64)
        if not np.array_equal(candidate_fold, reference["fold"].to_numpy(dtype=np.int64)):
            raise ValueError(f"Fold assignments differ for {name}")
        predictions.append(aligned["prediction"].to_numpy(dtype=np.float64))

    fold = reference["fold"].to_numpy(dtype=np.int64)
    unique_folds = np.unique(fold)
    if len(unique_folds) < 2:
        raise ValueError("Cross-fitted blending requires at least two outer folds")
    # Grouped CV requires a time_id to live in exactly one validation fold.
    fold_per_time = reference.loc[:, ["time_id", "fold"]].drop_duplicates()
    if fold_per_time["time_id"].duplicated().any():
        raise ValueError("A time_id appears in more than one outer fold")

    return AlignedOOF(
        keys=reference.loc[:, KEY_COLUMNS].reset_index(drop=True),
        target=reference["target"].to_numpy(dtype=np.float64),
        fold=fold,
        predictions=np.column_stack(predictions),
        model_names=names,
    )


def _project_to_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto ``w >= 0, sum(w) = 1``."""

    vector = np.asarray(values, dtype=np.float64)
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    valid = ordered - cumulative / np.arange(1, len(vector) + 1) > 0
    if not np.any(valid):  # Defensive only; the simplex projection always exists.
        return np.full(len(vector), 1.0 / len(vector))
    rho = np.flatnonzero(valid)[-1]
    theta = cumulative[rho] / float(rho + 1)
    return np.maximum(vector - theta, 0.0)


def fit_simplex_rmspe_weights(
    y_true: np.ndarray | list[float],
    predictions: np.ndarray,
    *,
    max_iter: int = 20_000,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Fit non-negative, sum-to-one weights under the RMSPE objective.

    For two models, the constrained optimum has a closed form.  For three or
    more models, deterministic accelerated projected gradient solves the
    convex weighted least-squares problem on the probability simplex.
    """

    target = validate_positive_targets(y_true)
    matrix = np.asarray(predictions, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(target):
        raise ValueError("predictions must have shape (n_rows, n_models)")
    if matrix.shape[1] < 2:
        raise ValueError("At least two prediction columns are required")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("predictions contain NaN or infinite values")
    if max_iter < 1 or tolerance <= 0:
        raise ValueError("max_iter and tolerance must be positive")

    # Base learners are evaluated with non-negative clipping.  Clipping before
    # a simplex blend keeps the fitted and reported objectives identical.
    matrix = np.clip(matrix, 0.0, None)
    if matrix.shape[1] == 2:
        difference = (matrix[:, 0] - matrix[:, 1]) / target
        residual_from_second = (target - matrix[:, 1]) / target
        denominator = float(difference @ difference)
        if denominator <= np.finfo(np.float64).eps:
            return np.array([0.5, 0.5], dtype=np.float64)
        first_weight = float(
            np.clip((difference @ residual_from_second) / denominator, 0.0, 1.0)
        )
        return np.array([first_weight, 1.0 - first_weight], dtype=np.float64)

    design = matrix / target[:, None]
    hessian = 2.0 * (design.T @ design) / len(target)
    linear = 2.0 * design.mean(axis=0)
    lipschitz = float(np.linalg.eigvalsh(hessian).max())
    if not np.isfinite(lipschitz) or lipschitz <= np.finfo(np.float64).eps:
        return np.full(matrix.shape[1], 1.0 / matrix.shape[1])

    weights = np.full(matrix.shape[1], 1.0 / matrix.shape[1])
    accelerated = weights.copy()
    momentum = 1.0
    for _ in range(max_iter):
        gradient = hessian @ accelerated - linear
        updated = _project_to_simplex(accelerated - gradient / lipschitz)
        if np.max(np.abs(updated - weights)) <= tolerance:
            weights = updated
            break
        next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
        accelerated = updated + ((momentum - 1.0) / next_momentum) * (
            updated - weights
        )
        weights = updated
        momentum = next_momentum
    return _project_to_simplex(weights)


def posthoc_oof_blend(
    aligned: AlignedOOF,
    *,
    bootstrap_resamples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 2021,
) -> PosthocOOFBlendResult:
    """Evaluate a post-hoc fold-exclusive blend.

    The held-out fold's targets are not read directly when fitting its weight,
    but indirect dependence through the other folds' base-model predictions is
    possible.  Scores from this function are descriptive and must not be
    reported as strict nested-CV estimates.
    """

    target = validate_positive_targets(aligned.target)
    component_predictions = np.asarray(aligned.predictions, dtype=np.float64)
    if component_predictions.shape != (len(target), len(aligned.model_names)):
        raise ValueError("AlignedOOF prediction shape does not match its model names")
    if len(aligned.fold) != len(target) or len(aligned.keys) != len(target):
        raise ValueError("AlignedOOF arrays have inconsistent lengths")
    if bootstrap_resamples < 2:
        raise ValueError("bootstrap_resamples must be at least 2")

    component_predictions = np.clip(component_predictions, 0.0, None)
    blend_oof = np.full(len(target), np.nan, dtype=np.float64)
    weight_records: list[dict[str, object]] = []
    metric_records: list[dict[str, object]] = []

    for fold_id in np.unique(aligned.fold):
        heldout = aligned.fold == fold_id
        meta_train = ~heldout
        if not np.any(heldout) or not np.any(meta_train):
            raise ValueError(f"Fold {fold_id} has an empty train or held-out partition")
        weights = fit_simplex_rmspe_weights(
            target[meta_train], component_predictions[meta_train]
        )
        heldout_prediction = np.clip(
            component_predictions[heldout] @ weights, 0.0, None
        )
        blend_oof[heldout] = heldout_prediction

        for model_name, weight in zip(aligned.model_names, weights, strict=True):
            weight_records.append(
                {
                    "fold": int(fold_id),
                    "model": model_name,
                    "weight": float(weight),
                    "meta_train_rows": int(meta_train.sum()),
                    "heldout_rows": int(heldout.sum()),
                }
            )
        metric_records.append(
            {
                "fold": int(fold_id),
                "model": "blend",
                "rmspe": rmspe(target[heldout], heldout_prediction),
                "rows": int(heldout.sum()),
            }
        )
        for column, model_name in enumerate(aligned.model_names):
            metric_records.append(
                {
                    "fold": int(fold_id),
                    "model": model_name,
                    "rmspe": rmspe(
                        target[heldout], component_predictions[heldout, column]
                    ),
                    "rows": int(heldout.sum()),
                }
            )

    if not np.all(np.isfinite(blend_oof)):
        raise RuntimeError("Cross-fitted blend failed to cover every OOF row")

    groups = aligned.keys["time_id"].to_numpy()
    prediction_by_model = {
        "blend": blend_oof,
        **{
            name: component_predictions[:, column]
            for column, name in enumerate(aligned.model_names)
        },
    }
    model_records: list[dict[str, object]] = []
    for offset, (model_name, prediction) in enumerate(prediction_by_model.items()):
        interval = grouped_bootstrap_rmspe(
            target,
            prediction,
            groups,
            n_resamples=bootstrap_resamples,
            confidence=confidence,
            seed=seed + offset,
        )
        model_records.append(
            {
                "model": model_name,
                "rmspe": interval.estimate,
                "ci_lower": interval.lower,
                "ci_upper": interval.upper,
                "confidence": interval.confidence,
                "bootstrap_resamples": interval.n_resamples,
            }
        )

    comparison_records: list[dict[str, object]] = []
    for offset, (model_name, prediction) in enumerate(
        zip(aligned.model_names, component_predictions.T, strict=True)
    ):
        difference = grouped_bootstrap_rmspe_difference(
            target,
            blend_oof,
            prediction,
            groups,
            n_resamples=bootstrap_resamples,
            confidence=confidence,
            seed=seed + offset,
        )
        comparison_records.append(
            {
                "candidate": "blend",
                "reference": model_name,
                "rmspe_difference": difference.estimate,
                "ci_lower": difference.lower,
                "ci_upper": difference.upper,
                "confidence": difference.confidence,
                "bootstrap_resamples": difference.n_resamples,
            }
        )

    model_summary = pd.DataFrame(model_records).sort_values("rmspe").reset_index(drop=True)
    paired_comparisons = pd.DataFrame(comparison_records)
    fold_metrics = pd.DataFrame(metric_records).sort_values(
        ["fold", "model"]
    ).reset_index(drop=True)
    fold_weights = pd.DataFrame(weight_records).sort_values(
        ["fold", "model"]
    ).reset_index(drop=True)
    deployment_weight_values = fit_simplex_rmspe_weights(
        target, component_predictions
    )
    deployment_weights = pd.DataFrame(
        {
            "model": aligned.model_names,
            "weight": deployment_weight_values,
            "fit_rows": len(target),
        }
    )
    blend_row = model_summary.loc[model_summary["model"] == "blend"].iloc[0]
    summary: dict[str, object] = {
        "n_rows": len(target),
        "n_folds": int(len(np.unique(aligned.fold))),
        "n_models": len(aligned.model_names),
        "component_models": list(aligned.model_names),
        "pooled_oof_rmspe": float(blend_row["rmspe"]),
        "group_bootstrap_ci_lower": float(blend_row["ci_lower"]),
        "group_bootstrap_ci_upper": float(blend_row["ci_upper"]),
        "confidence": confidence,
        "bootstrap_resamples": bootstrap_resamples,
        "weight_constraint": "nonnegative_sum_to_one",
        "method": "posthoc_oof_blend",
        "validation_claim": "descriptive_posthoc_not_strict_nested",
        "meta_validation": "leave_one_outer_fold_out_posthoc",
        "indirect_dependency_warning": (
            "Other-fold OOF predictions may depend indirectly on the nominally "
            "held-out fold because their base models can be trained on it; this "
            "estimate is not unbiased nested CV."
        ),
        "deployment_weight_fit": (
            "all OOF rows; for retrained base models on future unseen data only"
        ),
        "deployment_weights": {
            model: float(weight)
            for model, weight in zip(
                aligned.model_names, deployment_weight_values, strict=True
            )
        },
    }
    return PosthocOOFBlendResult(
        oof_predictions=blend_oof,
        fold_weights=fold_weights,
        deployment_weights=deployment_weights,
        fold_metrics=fold_metrics,
        model_summary=model_summary,
        paired_comparisons=paired_comparisons,
        summary=summary,
    )


# Backward-compatible name for callers that imported the original result type.
CrossFittedBlendResult = PosthocOOFBlendResult


def cross_fitted_blend(
    aligned: AlignedOOF,
    *,
    bootstrap_resamples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 2021,
) -> PosthocOOFBlendResult:
    """Compatibility wrapper for :func:`posthoc_oof_blend`.

    Despite its historical name, this is not a strict cross-fitted meta-model.
    The returned metadata always labels the method ``posthoc_oof_blend`` and
    records the indirect-dependency limitation.
    """

    return posthoc_oof_blend(
        aligned,
        bootstrap_resamples=bootstrap_resamples,
        confidence=confidence,
        seed=seed,
    )


__all__ = [
    "AlignedOOF",
    "CrossFittedBlendResult",
    "PosthocOOFBlendResult",
    "align_oof_frames",
    "cross_fitted_blend",
    "fit_simplex_rmspe_weights",
    "posthoc_oof_blend",
]
