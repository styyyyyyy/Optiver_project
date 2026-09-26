#!/usr/bin/env python3
"""Generate one leakage-isolated calibration bundle for nested blending.

The requested outer fold is never used to fit either component model or the
blend weight.  Inside the corresponding outer-training partition, one
grouped split is used as a calibration holdout.  Component models are fitted
only on the complementary inner-fit groups and predict that holdout.  The
saved manifest makes those partitions machine-verifiable by
``evaluate_nested_blend.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import sklearn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from optiver.blending import align_oof_frames  # noqa: E402
from optiver.data import load_feature_table  # noqa: E402
from optiver.feature_sets import get_feature_set  # noqa: E402
from optiver.nested_blending import (  # noqa: E402
    SCHEMA_VERSION,
    hash_group_ids,
    sha256_file,
)
from optiver.training import LightGBMConfig, run_lightgbm_cv  # noqa: E402
from optiver.validation import build_group_folds  # noqa: E402
from scripts.run_experiments import (  # noqa: E402
    CLUSTER_SOURCE_COLUMNS,
    FEATURE_CLUSTER_COLUMNS,
    _feature_provider,
    _inner_split_factory,
)


MODEL_NAMES = ("lightgbm", "mlp")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, required=True)
    parser.add_argument("--lightgbm-oof", type=Path, required=True)
    parser.add_argument("--mlp-oof", type=Path, required=True)
    parser.add_argument(
        "--mlp-calibration-run",
        type=Path,
        required=True,
        help="Completed output from run_nested_calibration_mlp_fold.py",
    )
    parser.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--num-threads", type=int, default=2)
    return parser.parse_args()


def _resolve_oof(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "oof_predictions.parquet"
    if not resolved.is_file():
        raise FileNotFoundError(f"OOF parquet does not exist: {resolved}")
    return resolved


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_hashes() -> dict[str, str]:
    sources = [
        PROJECT_ROOT / "pyproject.toml",
        Path(__file__).resolve(),
        PROJECT_ROOT / "scripts" / "run_experiments.py",
        *sorted((PROJECT_ROOT / "src" / "optiver").glob("*.py")),
    ]
    return {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in sources
        if path.is_file()
    }


def _load_outer_oof(lightgbm_path: Path, mlp_path: Path):
    columns = ["stock_id", "time_id", "target", "fold", "prediction"]
    frames = {
        "lightgbm": pd.read_parquet(lightgbm_path, columns=columns),
        "mlp": pd.read_parquet(mlp_path, columns=columns),
    }
    aligned = align_oof_frames(frames)
    if aligned.model_names != MODEL_NAMES:
        raise RuntimeError(
            f"Unexpected aligned model order: {aligned.model_names!r}"
        )
    folds = sorted(int(value) for value in np.unique(aligned.fold))
    if folds != list(range(5)):
        raise ValueError(f"Expected outer folds 0..4, received {folds}")
    return aligned


def _assert_frame_matches_outer_oof(frame: pd.DataFrame, aligned: Any) -> None:
    reference = aligned.keys.copy()
    reference["target"] = aligned.target
    observed = frame[["stock_id", "time_id", "target"]]
    if len(observed) != len(reference):
        raise ValueError("Feature table and outer OOF row counts differ")
    merged = reference.merge(
        observed,
        on=["stock_id", "time_id"],
        suffixes=("_oof", "_features"),
        validate="one_to_one",
    )
    if len(merged) != len(reference):
        raise ValueError("Feature table and outer OOF key coverage differs")
    if not np.array_equal(
        merged["target_oof"].to_numpy(dtype=np.float64),
        merged["target_features"].to_numpy(dtype=np.float64),
    ):
        raise ValueError("Feature table and outer OOF targets differ")


def _group_fold_map(aligned: Any) -> dict[int, int]:
    mapping = aligned.keys[["time_id"]].copy()
    mapping["fold"] = aligned.fold
    per_group = mapping.drop_duplicates()
    if per_group["time_id"].duplicated().any():
        raise ValueError("An outer time_id appears in more than one fold")
    return {
        int(row.time_id): int(row.fold)
        for row in per_group.itertuples(index=False)
    }


def _long_calibration_frame(
    calibration: pd.DataFrame,
    *,
    outer_fold: int,
    lightgbm_prediction: np.ndarray,
    mlp_prediction: np.ndarray,
) -> pd.DataFrame:
    keys = calibration[["stock_id", "time_id", "target"]].reset_index(drop=True)
    records: list[pd.DataFrame] = []
    for model, prediction in (
        ("lightgbm", lightgbm_prediction),
        ("mlp", mlp_prediction),
    ):
        if len(prediction) != len(keys) or not np.all(np.isfinite(prediction)):
            raise RuntimeError(f"Invalid {model} calibration predictions")
        component = keys.copy()
        component["outer_fold"] = outer_fold
        component["model"] = model
        component["prediction"] = prediction
        records.append(component)
    return pd.concat(records, ignore_index=True)


def _load_mlp_calibration_component(
    supplied_path: Path,
    *,
    outer_fold: int,
    expected_frame: pd.DataFrame,
    fit_groups: tuple[int, ...],
    calibration_groups: tuple[int, ...],
    outer_train_groups: tuple[int, ...],
    outer_validation_groups: tuple[int, ...],
    outer_oof_path: Path,
) -> tuple[np.ndarray, dict[str, Any], Path]:
    run_dir = supplied_path.expanduser().resolve()
    completion_path = run_dir / "COMPLETED.json"
    manifest_path = run_dir / "component_manifest.json"
    if not completion_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"MLP calibration component is incomplete: {run_dir}"
        )
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "completed":
        raise ValueError("MLP calibration component is not marked completed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "nested_calibration_component_v1":
        raise ValueError("Unsupported MLP calibration component schema")
    if manifest.get("model") != "mlp" or manifest.get("outer_fold") != outer_fold:
        raise ValueError("MLP component model/fold does not match requested fold")

    expected_groups = {
        "fit_groups": fit_groups,
        "calibration_groups": calibration_groups,
        "outer_train_groups": outer_train_groups,
        "outer_validation_groups": outer_validation_groups,
    }
    for name, expected in expected_groups.items():
        observed = tuple(sorted(int(x) for x in manifest.get(name, [])))
        if observed != expected:
            raise ValueError(f"MLP component {name} differs from current partition")
        claimed_hash = manifest.get("group_hashes", {}).get(f"{name}_sha256")
        if claimed_hash != hash_group_ids(expected):
            raise ValueError(f"MLP component {name} hash mismatch")
    if manifest.get("outer_prediction_file_sha256") != sha256_file(outer_oof_path):
        raise ValueError("MLP component is bound to a different outer OOF file")

    prediction_record = manifest.get("predictions", {})
    relative_path = prediction_record.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("MLP component prediction path is missing")
    prediction_path = (run_dir / relative_path).resolve()
    if not prediction_path.is_relative_to(run_dir) or not prediction_path.is_file():
        raise ValueError("MLP component prediction path is invalid")
    if prediction_record.get("sha256") != sha256_file(prediction_path):
        raise ValueError("MLP component prediction file hash mismatch")
    observed = pd.read_parquet(
        prediction_path,
        columns=["stock_id", "time_id", "target", "prediction"],
    )
    if observed.duplicated(["stock_id", "time_id"]).any():
        raise ValueError("MLP component contains duplicate prediction keys")
    expected = expected_frame[["stock_id", "time_id", "target"]].reset_index(
        drop=True
    )
    aligned = expected.merge(
        observed,
        on=["stock_id", "time_id"],
        how="left",
        suffixes=("_expected", "_observed"),
        validate="one_to_one",
        sort=False,
    )
    if len(aligned) != len(expected) or aligned["prediction"].isna().any():
        raise ValueError("MLP component prediction key coverage differs")
    if len(observed) != len(expected):
        raise ValueError("MLP component contains extra prediction keys")
    if not np.array_equal(
        aligned["target_expected"].to_numpy(dtype=np.float64),
        aligned["target_observed"].to_numpy(dtype=np.float64),
    ):
        raise ValueError("MLP component targets differ from calibration frame")
    prediction = aligned["prediction"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(prediction)):
        raise ValueError("MLP component predictions are non-finite")
    return prediction, manifest, run_dir


def _execute(args: argparse.Namespace, output_dir: Path) -> None:
    feature_path = args.features_path.expanduser().resolve()
    lightgbm_oof_path = _resolve_oof(args.lightgbm_oof)
    mlp_oof_path = _resolve_oof(args.mlp_oof)
    aligned = _load_outer_oof(lightgbm_oof_path, mlp_oof_path)

    schema = pq.ParquetFile(feature_path).schema_arrow.names
    lightgbm_features = get_feature_set("full_clean", schema, strict=True)
    needed = list(
        dict.fromkeys(
            [
                "stock_id",
                "time_id",
                "target",
                *lightgbm_features,
                *CLUSTER_SOURCE_COLUMNS,
                *FEATURE_CLUSTER_COLUMNS,
            ]
        )
    )
    frame = load_feature_table(feature_path, columns=needed)
    _assert_frame_matches_outer_oof(frame, aligned)

    group_to_fold = _group_fold_map(aligned)
    row_outer_fold = frame["time_id"].map(group_to_fold)
    if row_outer_fold.isna().any():
        raise ValueError("Feature table contains time_id values absent from outer OOF")
    outer_validation_mask = row_outer_fold.to_numpy(dtype=np.int16) == args.outer_fold
    if not np.any(outer_validation_mask):
        raise ValueError(f"Outer fold {args.outer_fold} has no rows")
    outer_train = frame.loc[~outer_validation_mask].reset_index(drop=True)

    calibration_seed = args.seed + 40_000 + args.outer_fold
    inner_fold = build_group_folds(
        outer_train[["time_id"]],
        group_col="time_id",
        n_splits=args.inner_splits,
        strategy="group_kfold",
        random_state=calibration_seed,
    )[0]
    fit_idx = inner_fold.train_idx
    calibration_idx = inner_fold.valid_idx
    fit_frame = outer_train.iloc[fit_idx]
    calibration_frame = outer_train.iloc[calibration_idx]
    fit_groups = tuple(sorted(int(x) for x in pd.unique(fit_frame["time_id"])))
    calibration_groups = tuple(
        sorted(int(x) for x in pd.unique(calibration_frame["time_id"]))
    )
    outer_train_groups = tuple(
        sorted(int(x) for x in pd.unique(outer_train["time_id"]))
    )
    outer_validation_groups = tuple(
        sorted(int(x) for x in pd.unique(frame.loc[outer_validation_mask, "time_id"]))
    )
    if set(fit_groups).intersection(calibration_groups):
        raise RuntimeError("Inner fit and calibration groups overlap")
    if set(fit_groups).union(calibration_groups) != set(outer_train_groups):
        raise RuntimeError("Inner fit and calibration do not partition outer train")

    model_seed = args.seed + args.outer_fold
    lightgbm_params = {
        "objective": "regression",
        "metric": "None",
        "boosting_type": "gbdt",
        "num_leaves": 256,
        "learning_rate": 0.05,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 20,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "verbosity": -1,
        "num_threads": args.num_threads,
        "deterministic": True,
        "force_col_wise": True,
    }
    lightgbm_config = LightGBMConfig(
        params=lightgbm_params,
        max_boost_rounds=2_000,
        early_stopping_rounds=100,
        fixed_boost_rounds=90,
        seed=model_seed,
    )
    lightgbm_result = run_lightgbm_cv(
        outer_train,
        [(fit_idx, calibration_idx)],
        feature_provider=_feature_provider(
            lightgbm_features,
            include_stock_id=True,
            include_stock_target_stats=False,
            inner_splits=args.inner_splits,
            cluster_mode="feature",
            n_clusters=7,
            seed=model_seed,
        ),
        inner_split_factory=_inner_split_factory(args.inner_splits),
        config=lightgbm_config,
        model_dir=output_dir / "models" / "lightgbm",
        allow_partial_oof=True,
    )

    lightgbm_prediction = lightgbm_result.oof_predictions[calibration_idx]
    mlp_prediction, mlp_component, mlp_component_dir = (
        _load_mlp_calibration_component(
            args.mlp_calibration_run,
            outer_fold=args.outer_fold,
            expected_frame=calibration_frame,
            fit_groups=fit_groups,
            calibration_groups=calibration_groups,
            outer_train_groups=outer_train_groups,
            outer_validation_groups=outer_validation_groups,
            outer_oof_path=mlp_oof_path,
        )
    )
    long_frame = _long_calibration_frame(
        calibration_frame,
        outer_fold=args.outer_fold,
        lightgbm_prediction=lightgbm_prediction,
        mlp_prediction=mlp_prediction,
    )
    prediction_path = output_dir / "calibration_predictions.parquet"
    long_frame.to_parquet(prediction_path, index=False)

    lightgbm_result.fold_metrics.to_csv(
        output_dir / "lightgbm_fold_metrics.csv", index=False
    )
    lightgbm_result.importance.to_csv(
        output_dir / "lightgbm_feature_importance.csv", index=False
    )
    _json_dump(output_dir / "lightgbm_summary.json", lightgbm_result.summary)
    for name in (
        "mlp_fold_metrics.csv",
        "mlp_training_history.csv",
        "mlp_summary.json",
        "component_manifest.json",
    ):
        source = mlp_component_dir / name
        if source.is_file():
            target_name = (
                "mlp_component_manifest.json"
                if name == "component_manifest.json"
                else name
            )
            shutil.copy2(source, output_dir / target_name)
    mlp_models = mlp_component_dir / "models"
    if mlp_models.is_dir():
        shutil.copytree(mlp_models, output_dir / "models" / "mlp")

    source_hashes = _source_hashes()
    lightgbm_config_record = {
        "base_features": lightgbm_features,
        "include_stock_id": True,
        "cluster_mode": "feature",
        "n_clusters": 7,
        "fixed_boost_rounds": 90,
        "params": lightgbm_params,
        "seed": model_seed,
    }
    mlp_config_record = mlp_component["model_config"]
    model_records: dict[str, dict[str, object]] = {}
    for model_name, config_record, outer_path in (
        ("lightgbm", lightgbm_config_record, lightgbm_oof_path),
        ("mlp", mlp_config_record, mlp_oof_path),
    ):
        model_records[model_name] = {
            "fit_groups": list(fit_groups),
            "outer_prediction_fit_groups": list(outer_train_groups),
            "training_scope": "fit_groups_only",
            "training_seed": model_seed,
            "model_config": config_record,
            "model_config_sha256": _canonical_sha256(config_record),
            "outer_prediction_file": str(outer_path),
            "outer_prediction_file_sha256": sha256_file(outer_path),
            "source_files_sha256": (
                source_hashes
                if model_name == "lightgbm"
                else mlp_component["source_files_sha256"]
            ),
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outer_fold": args.outer_fold,
        "group_column": "time_id",
        "outer_train_groups": list(outer_train_groups),
        "outer_validation_groups": list(outer_validation_groups),
        "calibration_groups": list(calibration_groups),
        "models": model_records,
        "group_hashes": {
            "outer_train_groups_sha256": hash_group_ids(outer_train_groups),
            "outer_validation_groups_sha256": hash_group_ids(
                outer_validation_groups
            ),
            "calibration_groups_sha256": hash_group_ids(calibration_groups),
            "fit_groups_sha256_by_model": {
                model: hash_group_ids(fit_groups) for model in MODEL_NAMES
            },
            "outer_prediction_fit_groups_sha256_by_model": {
                model: hash_group_ids(outer_train_groups)
                for model in MODEL_NAMES
            },
        },
        "predictions": {
            "path": prediction_path.name,
            "sha256": sha256_file(prediction_path),
            "rows": int(len(long_frame)),
            "models": list(MODEL_NAMES),
        },
        "partition_protocol": {
            "name": "outer_train_grouped_calibration_holdout",
            "calibration_split": "first fold of five seeded grouped folds",
            "calibration_seed": calibration_seed,
            "outer_validation_access": "predictions_and_labels_not_used_by_calibration_training",
        },
        "mlp_component": {
            "path": str(mlp_component_dir),
            "manifest_sha256": sha256_file(
                mlp_component_dir / "component_manifest.json"
            ),
            "completion_sha256": sha256_file(
                mlp_component_dir / "COMPLETED.json"
            ),
        },
        "data": {
            "path": str(feature_path),
            "sha256": sha256_file(feature_path),
            "outer_train_rows": int(len(outer_train)),
            "fit_rows": int(len(fit_idx)),
            "calibration_rows": int(len(calibration_idx)),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
    }
    _json_dump(output_dir / "calibration_manifest.json", manifest)


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    status_path = output_dir / "status.json"
    _json_dump(status_path, {"status": "running", "started_at_utc": started_at})
    try:
        _execute(args, output_dir)
    except BaseException as error:
        _json_dump(
            status_path,
            {
                "status": "failed",
                "started_at_utc": started_at,
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    else:
        completion = {
            "status": "completed",
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "calibration_manifest_sha256": sha256_file(
                output_dir / "calibration_manifest.json"
            ),
        }
        _json_dump(output_dir / "COMPLETED.json", completion)
        _json_dump(status_path, completion)


if __name__ == "__main__":
    main()
