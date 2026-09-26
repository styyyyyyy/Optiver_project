"""Metrics and simple baselines for realized-volatility experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ArrayLike = np.ndarray | list[float]


def _as_1d_float(values: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape={array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def validate_positive_targets(y_true: ArrayLike) -> np.ndarray:
    """Return targets as float64 after enforcing the RMSPE domain."""

    targets = _as_1d_float(y_true, name="y_true")
    if np.any(targets <= 0):
        count = int(np.count_nonzero(targets <= 0))
        raise ValueError(f"RMSPE requires strictly positive targets; found {count}")
    return targets


def rmspe(y_true: ArrayLike, y_pred: ArrayLike, *, clip_predictions: bool = True) -> float:
    """Root mean squared percentage error with explicit input validation."""

    targets = validate_positive_targets(y_true)
    predictions = _as_1d_float(y_pred, name="y_pred")
    if targets.shape != predictions.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; got "
            f"{targets.shape} and {predictions.shape}"
        )
    if clip_predictions:
        predictions = np.clip(predictions, 0.0, None)
    return float(np.sqrt(np.mean(np.square((targets - predictions) / targets))))


def rmspe_weights(
    y_true: ArrayLike,
    *,
    normalize: bool = True,
    clip_quantile: float | None = None,
) -> np.ndarray:
    """Return ``1 / y**2`` weights used to align L2 training with RMSPE.

    Mean normalization leaves the data-fit optimum unchanged while keeping the
    regularization scale interpretable. Optional upper-tail clipping is only a
    robustness experiment and changes the target objective.
    """

    targets = validate_positive_targets(y_true)
    weights = np.reciprocal(np.square(targets))
    if clip_quantile is not None:
        if not 0.0 < clip_quantile <= 1.0:
            raise ValueError("clip_quantile must be in (0, 1]")
        upper = float(np.quantile(weights, clip_quantile))
        weights = np.minimum(weights, upper)
    if normalize:
        weights = weights / weights.mean()
    return weights


def optimal_rmspe_constant(y_train: ArrayLike) -> float:
    """Fit the constant forecast minimizing RMSPE on the training sample."""

    targets = validate_positive_targets(y_train)
    return float(np.reciprocal(targets).sum() / np.reciprocal(targets**2).sum())


def fit_rmspe_scale(y_train: ArrayLike, signal_train: ArrayLike) -> float:
    """Fit ``prediction = scale * signal`` under the RMSPE objective."""

    targets = validate_positive_targets(y_train)
    signal = _as_1d_float(signal_train, name="signal_train")
    if signal.shape != targets.shape:
        raise ValueError("signal_train and y_train must have the same shape")
    denominator = np.sum(np.square(signal / targets))
    if denominator <= 0:
        raise ValueError("signal_train has no non-zero information")
    numerator = np.sum(signal / targets)
    return float(numerator / denominator)


def lgb_rmspe(y_pred: np.ndarray, dataset: object) -> tuple[str, float, bool]:
    """LightGBM custom evaluation callback using the reported clipping rule."""

    y_true = dataset.get_label()  # type: ignore[attr-defined]
    return "rmspe", rmspe(y_true, y_pred, clip_predictions=True), False


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    n_resamples: int


@dataclass(frozen=True)
class BootstrapDifference:
    """Paired interval for ``RMSPE(candidate) - RMSPE(reference)``."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    n_resamples: int


def grouped_bootstrap_rmspe(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    groups: np.ndarray | list[int],
    *,
    n_resamples: int = 1_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapInterval:
    """Bootstrap RMSPE by resampling complete groups, not individual rows."""

    targets = validate_positive_targets(y_true)
    predictions = _as_1d_float(y_pred, name="y_pred")
    group_array = np.asarray(groups)
    if targets.shape != predictions.shape or targets.shape != group_array.shape:
        raise ValueError("y_true, y_pred, and groups must have identical shapes")
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    predictions = np.clip(predictions, 0.0, None)
    squared_percentage_error = np.square((targets - predictions) / targets)
    unique_groups, inverse = np.unique(group_array, return_inverse=True)
    group_error_sum = np.bincount(inverse, weights=squared_percentage_error)
    group_count = np.bincount(inverse)

    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0,
        len(unique_groups),
        size=(n_resamples, len(unique_groups)),
    )
    sampled_error = group_error_sum[draws].sum(axis=1)
    sampled_count = group_count[draws].sum(axis=1)
    estimates = np.sqrt(sampled_error / sampled_count)

    alpha = 1.0 - confidence
    lower, upper = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapInterval(
        estimate=rmspe(targets, predictions),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        n_resamples=n_resamples,
    )


def grouped_bootstrap_rmspe_difference(
    y_true: ArrayLike,
    candidate_pred: ArrayLike,
    reference_pred: ArrayLike,
    groups: np.ndarray | list[int],
    *,
    n_resamples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapDifference:
    """Paired grouped bootstrap for a model-to-model RMSPE difference.

    Negative values favor the candidate. Complete groups are sampled once and
    the identical draws are used for both models.
    """

    targets = validate_positive_targets(y_true)
    candidate = _as_1d_float(candidate_pred, name="candidate_pred")
    reference = _as_1d_float(reference_pred, name="reference_pred")
    group_array = np.asarray(groups)
    if not (
        targets.shape == candidate.shape == reference.shape == group_array.shape
    ):
        raise ValueError("targets, both predictions, and groups must have identical shapes")
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    candidate = np.clip(candidate, 0.0, None)
    reference = np.clip(reference, 0.0, None)
    candidate_error = np.square((targets - candidate) / targets)
    reference_error = np.square((targets - reference) / targets)
    unique_groups, inverse = np.unique(group_array, return_inverse=True)
    candidate_sum = np.bincount(inverse, weights=candidate_error)
    reference_sum = np.bincount(inverse, weights=reference_error)
    group_count = np.bincount(inverse)

    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0,
        len(unique_groups),
        size=(n_resamples, len(unique_groups)),
    )
    sampled_count = group_count[draws].sum(axis=1)
    candidate_scores = np.sqrt(candidate_sum[draws].sum(axis=1) / sampled_count)
    reference_scores = np.sqrt(reference_sum[draws].sum(axis=1) / sampled_count)
    differences = candidate_scores - reference_scores
    alpha = 1.0 - confidence
    lower, upper = np.quantile(differences, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapDifference(
        estimate=rmspe(targets, candidate) - rmspe(targets, reference),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        n_resamples=n_resamples,
    )
