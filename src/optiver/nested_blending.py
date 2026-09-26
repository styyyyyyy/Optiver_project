"""Strict nested calibration and evaluation for convex model blends.

For every outer fold, blend weights are fitted on a dedicated calibration
partition inside that fold's outer-training data.  Each component prediction
on the calibration partition must come from a model trained only on the
disjoint inner-fit partition.  A machine-verifiable provenance manifest binds
the group partitions and prediction file to the evaluation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .blending import AlignedOOF, fit_simplex_rmspe_weights
from .metrics import (
    grouped_bootstrap_rmspe,
    grouped_bootstrap_rmspe_difference,
    rmspe,
    validate_positive_targets,
)


SCHEMA_VERSION = "nested_calibration_v1"
CALIBRATION_COLUMNS = (
    "stock_id",
    "time_id",
    "target",
    "outer_fold",
    "model",
    "prediction",
)


@dataclass(frozen=True)
class NestedCalibrationFold:
    """Validated calibration predictions and their group provenance."""

    outer_fold: int
    keys: pd.DataFrame
    target: np.ndarray
    predictions: np.ndarray
    model_names: tuple[str, ...]
    outer_train_groups: tuple[int, ...]
    outer_validation_groups: tuple[int, ...]
    calibration_groups: tuple[int, ...]
    fit_groups_by_model: Mapping[str, tuple[int, ...]]
    outer_prediction_fit_groups_by_model: Mapping[str, tuple[int, ...]]
    group_hashes: Mapping[str, Any]
    manifest_sha256: str
    prediction_file_sha256: str | None
    source_manifest: Mapping[str, Any]


@dataclass(frozen=True)
class NestedBlendResult:
    """Strict nested-blend evaluation artifacts."""

    oof_predictions: np.ndarray
    fold_weights: pd.DataFrame
    fold_metrics: pd.DataFrame
    model_summary: pd.DataFrame
    paired_comparisons: pd.DataFrame
    summary: dict[str, object]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for an artifact file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_group_ids(groups: Sequence[int]) -> str:
    """Hash a canonical, sorted set of integer group identifiers."""

    normalized = _normalize_groups(groups, name="groups", allow_empty=True)
    return hashlib.sha256(_canonical_json(list(normalized))).hexdigest()


def _normalize_groups(
    raw_groups: Sequence[int] | object,
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if isinstance(raw_groups, (str, bytes)) or not isinstance(raw_groups, Sequence):
        raise ValueError(f"{name} must be a JSON array of integer group IDs")
    groups: list[int] = []
    for value in raw_groups:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must contain only integer group IDs")
        groups.append(int(value))
    if len(set(groups)) != len(groups):
        raise ValueError(f"{name} contains duplicate group IDs")
    if not groups and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    return tuple(sorted(groups))


def _required_mapping(
    value: object, *, name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _verify_hash(*, name: str, groups: tuple[int, ...], claimed: object) -> None:
    expected = hash_group_ids(groups)
    if not isinstance(claimed, str) or claimed != expected:
        raise ValueError(
            f"{name} hash mismatch: expected {expected}, received {claimed!r}"
        )


def _actual_outer_groups(
    outer_oof: AlignedOOF, outer_fold: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    time_ids = outer_oof.keys["time_id"].to_numpy()
    if not np.issubdtype(time_ids.dtype, np.integer):
        try:
            converted = time_ids.astype(np.int64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Outer OOF time_id values must be integers") from exc
        if not np.array_equal(converted.astype(time_ids.dtype), time_ids):
            raise ValueError("Outer OOF time_id values must be integers")
        time_ids = converted
    heldout = outer_oof.fold == outer_fold
    if not np.any(heldout):
        raise ValueError(f"Outer fold {outer_fold} does not exist in OOF data")
    validation_groups = tuple(sorted(int(x) for x in np.unique(time_ids[heldout])))
    train_groups = tuple(sorted(int(x) for x in np.unique(time_ids[~heldout])))
    return train_groups, validation_groups


def _parse_manifest_partitions(
    manifest: Mapping[str, Any],
    *,
    outer_oof: AlignedOOF,
) -> tuple[
    int,
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    dict[str, tuple[int, ...]],
    dict[str, tuple[int, ...]],
    Mapping[str, Any],
]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION!r}; "
            f"received {manifest.get('schema_version')!r}"
        )
    if manifest.get("group_column") != "time_id":
        raise ValueError("group_column must be 'time_id'")
    raw_outer_fold = manifest.get("outer_fold")
    if isinstance(raw_outer_fold, bool) or not isinstance(
        raw_outer_fold, (int, np.integer)
    ):
        raise ValueError("outer_fold must be an integer")
    outer_fold = int(raw_outer_fold)

    outer_train = _normalize_groups(
        manifest.get("outer_train_groups"), name="outer_train_groups"
    )
    outer_validation = _normalize_groups(
        manifest.get("outer_validation_groups"),
        name="outer_validation_groups",
    )
    calibration = _normalize_groups(
        manifest.get("calibration_groups"), name="calibration_groups"
    )
    actual_train, actual_validation = _actual_outer_groups(outer_oof, outer_fold)
    if outer_train != actual_train:
        raise ValueError(
            f"outer_train_groups do not match OOF complement for fold {outer_fold}"
        )
    if outer_validation != actual_validation:
        raise ValueError(
            f"outer_validation_groups do not match OOF fold {outer_fold}"
        )
    if set(outer_train).intersection(outer_validation):
        raise ValueError("outer train and validation groups overlap")

    manifest_models = _required_mapping(manifest.get("models"), name="models")
    if set(manifest_models) != set(outer_oof.model_names):
        raise ValueError(
            "Calibration manifest model labels differ from outer OOF models: "
            f"expected={sorted(outer_oof.model_names)}, "
            f"received={sorted(manifest_models)}"
        )
    fit_groups_by_model: dict[str, tuple[int, ...]] = {}
    outer_fit_groups_by_model: dict[str, tuple[int, ...]] = {}
    for model_name in outer_oof.model_names:
        record = _required_mapping(
            manifest_models[model_name], name=f"models.{model_name}"
        )
        if record.get("training_scope") != "fit_groups_only":
            raise ValueError(
                f"models.{model_name}.training_scope must be 'fit_groups_only'"
            )
        fit_groups = _normalize_groups(
            record.get("fit_groups"), name=f"models.{model_name}.fit_groups"
        )
        outer_fit_groups = _normalize_groups(
            record.get("outer_prediction_fit_groups"),
            name=f"models.{model_name}.outer_prediction_fit_groups",
        )
        if set(fit_groups).intersection(calibration):
            raise ValueError(
                f"{model_name} fit groups overlap calibration groups for fold {outer_fold}"
            )
        if set(fit_groups).intersection(outer_validation):
            raise ValueError(
                f"{model_name} fit groups overlap outer validation groups"
            )
        if set(calibration).intersection(outer_validation):
            raise ValueError("Calibration groups overlap outer validation groups")
        if set(fit_groups).union(calibration) != set(outer_train):
            raise ValueError(
                f"{model_name} fit and calibration groups must partition outer train"
            )
        if outer_fit_groups != outer_train:
            raise ValueError(
                f"{model_name} outer_prediction_fit_groups must equal outer train"
            )
        fit_groups_by_model[model_name] = fit_groups
        outer_fit_groups_by_model[model_name] = outer_fit_groups

    hashes = _required_mapping(manifest.get("group_hashes"), name="group_hashes")
    _verify_hash(
        name="outer_train_groups",
        groups=outer_train,
        claimed=hashes.get("outer_train_groups_sha256"),
    )
    _verify_hash(
        name="outer_validation_groups",
        groups=outer_validation,
        claimed=hashes.get("outer_validation_groups_sha256"),
    )
    _verify_hash(
        name="calibration_groups",
        groups=calibration,
        claimed=hashes.get("calibration_groups_sha256"),
    )
    fit_hashes = _required_mapping(
        hashes.get("fit_groups_sha256_by_model"),
        name="group_hashes.fit_groups_sha256_by_model",
    )
    outer_fit_hashes = _required_mapping(
        hashes.get("outer_prediction_fit_groups_sha256_by_model"),
        name="group_hashes.outer_prediction_fit_groups_sha256_by_model",
    )
    if set(fit_hashes) != set(outer_oof.model_names):
        raise ValueError("fit_groups hash model labels are incomplete")
    if set(outer_fit_hashes) != set(outer_oof.model_names):
        raise ValueError("outer_prediction_fit_groups hash model labels are incomplete")
    for model_name in outer_oof.model_names:
        _verify_hash(
            name=f"{model_name} fit_groups",
            groups=fit_groups_by_model[model_name],
            claimed=fit_hashes.get(model_name),
        )
        _verify_hash(
            name=f"{model_name} outer_prediction_fit_groups",
            groups=outer_fit_groups_by_model[model_name],
            claimed=outer_fit_hashes.get(model_name),
        )
    return (
        outer_fold,
        outer_train,
        outer_validation,
        calibration,
        fit_groups_by_model,
        outer_fit_groups_by_model,
        hashes,
    )


def validate_nested_calibration_frame(
    calibration_frame: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    outer_oof: AlignedOOF,
    prediction_file_sha256: str | None = None,
) -> NestedCalibrationFold:
    """Validate one outer fold's long-form calibration table and manifest."""

    (
        outer_fold,
        outer_train,
        outer_validation,
        calibration_groups,
        fit_groups_by_model,
        outer_fit_groups_by_model,
        group_hashes,
    ) = _parse_manifest_partitions(manifest, outer_oof=outer_oof)

    missing = set(CALIBRATION_COLUMNS).difference(calibration_frame.columns)
    if missing:
        raise ValueError(f"Calibration table is missing columns: {sorted(missing)}")
    frame = calibration_frame.loc[:, CALIBRATION_COLUMNS].copy()
    if frame.empty:
        raise ValueError("Calibration table must not be empty")
    if frame[["stock_id", "time_id", "model"]].isna().any(axis=None):
        raise ValueError("Calibration table contains null alignment keys")
    if frame.duplicated(["model", "stock_id", "time_id"]).any():
        raise ValueError("Calibration table contains duplicate model/key rows")
    if set(frame["model"].astype(str).unique()) != set(outer_oof.model_names):
        raise ValueError("Calibration table model labels differ from outer OOF models")
    fold_values = frame["outer_fold"].to_numpy()
    if not np.all(fold_values == outer_fold):
        raise ValueError(
            f"Calibration table outer_fold values must all equal {outer_fold}"
        )
    actual_calibration_groups = _normalize_groups(
        [int(value) for value in pd.unique(frame["time_id"])],
        name="calibration table time_id groups",
    )
    if actual_calibration_groups != calibration_groups:
        raise ValueError("Calibration table groups differ from calibration_groups")

    prediction_record = _required_mapping(
        manifest.get("predictions"), name="predictions"
    )
    claimed_prediction_hash = prediction_record.get("sha256")
    if not isinstance(claimed_prediction_hash, str):
        raise ValueError("predictions.sha256 must be present")
    if prediction_file_sha256 is not None and (
        prediction_file_sha256 != claimed_prediction_hash
    ):
        raise ValueError("Calibration prediction file hash mismatch")

    outer_reference = pd.DataFrame(
        {
            "stock_id": outer_oof.keys["stock_id"].to_numpy(),
            "time_id": outer_oof.keys["time_id"].to_numpy(),
            "target": outer_oof.target,
        }
    )
    expected = outer_reference.loc[
        outer_reference["time_id"].isin(calibration_groups)
    ].copy()
    expected_index = pd.MultiIndex.from_frame(expected[["stock_id", "time_id"]])
    if expected_index.has_duplicates:
        raise ValueError("Outer OOF contains duplicate keys in calibration groups")

    predictions: list[np.ndarray] = []
    for model_name in outer_oof.model_names:
        component = frame.loc[
            frame["model"].astype(str) == model_name,
            ["stock_id", "time_id", "target", "prediction"],
        ].copy()
        component_index = pd.MultiIndex.from_frame(
            component[["stock_id", "time_id"]]
        )
        missing_keys = expected_index.difference(component_index)
        extra_keys = component_index.difference(expected_index)
        if len(missing_keys) or len(extra_keys) or len(component) != len(expected):
            raise ValueError(
                f"Calibration key coverage differs for {model_name}: "
                f"missing={len(missing_keys)}, extra={len(extra_keys)}"
            )
        component = (
            component.set_index(["stock_id", "time_id"])
            .loc[expected_index]
            .reset_index()
        )
        target = component["target"].to_numpy(dtype=np.float64)
        if not np.array_equal(target, expected["target"].to_numpy(dtype=np.float64)):
            raise ValueError(f"Calibration targets differ for {model_name}")
        prediction = component["prediction"].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(prediction)):
            raise ValueError(f"Calibration predictions are non-finite for {model_name}")
        predictions.append(prediction)

    target = validate_positive_targets(expected["target"].to_numpy(dtype=np.float64))
    manifest_hash = hashlib.sha256(_canonical_json(manifest)).hexdigest()
    return NestedCalibrationFold(
        outer_fold=outer_fold,
        keys=expected[["stock_id", "time_id"]].reset_index(drop=True),
        target=target,
        predictions=np.column_stack(predictions),
        model_names=outer_oof.model_names,
        outer_train_groups=outer_train,
        outer_validation_groups=outer_validation,
        calibration_groups=calibration_groups,
        fit_groups_by_model=fit_groups_by_model,
        outer_prediction_fit_groups_by_model=outer_fit_groups_by_model,
        group_hashes=group_hashes,
        manifest_sha256=manifest_hash,
        prediction_file_sha256=prediction_file_sha256,
        source_manifest=dict(manifest),
    )


def load_nested_calibration_bundle(
    path: str | Path,
    *,
    outer_oof: AlignedOOF,
) -> NestedCalibrationFold:
    """Load and validate one ``nested_calibration_v1`` bundle."""

    supplied = Path(path).expanduser().resolve()
    manifest_path = supplied / "calibration_manifest.json" if supplied.is_dir() else supplied
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Calibration manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _required_mapping(manifest, name="calibration manifest")
    prediction_record = _required_mapping(
        manifest.get("predictions"), name="predictions"
    )
    relative_path = prediction_record.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("predictions.path must be a non-empty relative path")
    prediction_path = (manifest_path.parent / relative_path).resolve()
    if not prediction_path.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError("predictions.path must stay inside the calibration bundle")
    if not prediction_path.is_file():
        raise FileNotFoundError(
            f"Calibration prediction parquet does not exist: {prediction_path}"
        )
    prediction_hash = sha256_file(prediction_path)
    frame = pd.read_parquet(prediction_path, columns=list(CALIBRATION_COLUMNS))
    return validate_nested_calibration_frame(
        frame,
        manifest,
        outer_oof=outer_oof,
        prediction_file_sha256=prediction_hash,
    )


def nested_calibrated_blend(
    outer_oof: AlignedOOF,
    calibration_folds: Mapping[int, NestedCalibrationFold],
    *,
    bootstrap_resamples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 2021,
) -> NestedBlendResult:
    """Evaluate a strictly nested blend using fold-local calibration data."""

    if bootstrap_resamples < 2:
        raise ValueError("bootstrap_resamples must be at least 2")
    target = validate_positive_targets(outer_oof.target)
    component_predictions = np.asarray(outer_oof.predictions, dtype=np.float64)
    if component_predictions.shape != (
        len(target),
        len(outer_oof.model_names),
    ):
        raise ValueError("Outer OOF prediction shape does not match model names")
    if not np.all(np.isfinite(component_predictions)):
        raise ValueError("Outer OOF predictions contain non-finite values")
    component_predictions = np.clip(component_predictions, 0.0, None)
    expected_folds = {int(value) for value in np.unique(outer_oof.fold)}
    if set(calibration_folds) != expected_folds:
        raise ValueError(
            "Calibration bundles must cover every outer fold exactly once: "
            f"expected={sorted(expected_folds)}, "
            f"received={sorted(calibration_folds)}"
        )

    nested_oof = np.full(len(target), np.nan, dtype=np.float64)
    weight_records: list[dict[str, object]] = []
    metric_records: list[dict[str, object]] = []
    provenance_records: dict[str, object] = {}
    for outer_fold in sorted(expected_folds):
        calibration = calibration_folds[outer_fold]
        if calibration.outer_fold != outer_fold:
            raise ValueError("Calibration mapping key differs from bundle outer_fold")
        if calibration.model_names != outer_oof.model_names:
            raise ValueError(f"Model order differs in calibration fold {outer_fold}")
        heldout = outer_oof.fold == outer_fold
        weights = fit_simplex_rmspe_weights(
            calibration.target, calibration.predictions
        )
        prediction = np.clip(
            component_predictions[heldout] @ weights, 0.0, None
        )
        nested_oof[heldout] = prediction

        for model_name, weight in zip(outer_oof.model_names, weights, strict=True):
            weight_records.append(
                {
                    "fold": outer_fold,
                    "model": model_name,
                    "weight": float(weight),
                    "calibration_rows": len(calibration.target),
                    "calibration_groups": len(calibration.calibration_groups),
                    "fit_groups": len(calibration.fit_groups_by_model[model_name]),
                    "calibration_groups_sha256": hash_group_ids(
                        calibration.calibration_groups
                    ),
                    "fit_groups_sha256": hash_group_ids(
                        calibration.fit_groups_by_model[model_name]
                    ),
                    "manifest_sha256": calibration.manifest_sha256,
                }
            )
        metric_records.append(
            {
                "fold": outer_fold,
                "model": "blend",
                "rmspe": rmspe(target[heldout], prediction),
                "rows": int(heldout.sum()),
            }
        )
        for column, model_name in enumerate(outer_oof.model_names):
            metric_records.append(
                {
                    "fold": outer_fold,
                    "model": model_name,
                    "rmspe": rmspe(
                        target[heldout], component_predictions[heldout, column]
                    ),
                    "rows": int(heldout.sum()),
                }
            )
        provenance_records[str(outer_fold)] = {
            "manifest_sha256": calibration.manifest_sha256,
            "prediction_file_sha256": calibration.prediction_file_sha256,
            "outer_train_groups_sha256": hash_group_ids(
                calibration.outer_train_groups
            ),
            "outer_validation_groups_sha256": hash_group_ids(
                calibration.outer_validation_groups
            ),
            "calibration_groups_sha256": hash_group_ids(
                calibration.calibration_groups
            ),
            "fit_groups_sha256_by_model": {
                model: hash_group_ids(groups)
                for model, groups in calibration.fit_groups_by_model.items()
            },
        }

    if not np.all(np.isfinite(nested_oof)):
        raise RuntimeError("Strict nested blend did not cover every OOF row")

    groups = outer_oof.keys["time_id"].to_numpy()
    prediction_by_model = {
        "blend": nested_oof,
        **{
            name: component_predictions[:, column]
            for column, name in enumerate(outer_oof.model_names)
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
        zip(outer_oof.model_names, component_predictions.T, strict=True)
    ):
        difference = grouped_bootstrap_rmspe_difference(
            target,
            nested_oof,
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

    model_summary = (
        pd.DataFrame(model_records).sort_values("rmspe").reset_index(drop=True)
    )
    blend_row = model_summary.loc[model_summary["model"] == "blend"].iloc[0]
    summary: dict[str, object] = {
        "method": "strict_nested_calibration_blend",
        "validation_claim": "strict_nested_outer_oof",
        "n_rows": len(target),
        "n_folds": len(expected_folds),
        "n_models": len(outer_oof.model_names),
        "component_models": list(outer_oof.model_names),
        "pooled_oof_rmspe": float(blend_row["rmspe"]),
        "group_bootstrap_ci_lower": float(blend_row["ci_lower"]),
        "group_bootstrap_ci_upper": float(blend_row["ci_upper"]),
        "confidence": confidence,
        "bootstrap_resamples": bootstrap_resamples,
        "weight_constraint": "nonnegative_sum_to_one",
        "weight_fit": "fold-local inner calibration only",
        "uncertainty_caveat": (
            "Grouped bootstrap intervals are conditional on the fitted base "
            "models and fold-local calibration weights; they do not include "
            "model-training or calibration-selection uncertainty."
        ),
        "provenance": provenance_records,
    }
    return NestedBlendResult(
        oof_predictions=nested_oof,
        fold_weights=(
            pd.DataFrame(weight_records)
            .sort_values(["fold", "model"])
            .reset_index(drop=True)
        ),
        fold_metrics=(
            pd.DataFrame(metric_records)
            .sort_values(["fold", "model"])
            .reset_index(drop=True)
        ),
        model_summary=model_summary,
        paired_comparisons=pd.DataFrame(comparison_records),
        summary=summary,
    )


__all__ = [
    "CALIBRATION_COLUMNS",
    "SCHEMA_VERSION",
    "NestedBlendResult",
    "NestedCalibrationFold",
    "hash_group_ids",
    "load_nested_calibration_bundle",
    "nested_calibrated_blend",
    "sha256_file",
    "validate_nested_calibration_frame",
]
