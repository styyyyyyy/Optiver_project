#!/usr/bin/env python3
"""Train one MLP calibration component in a LightGBM-free process.

PyTorch and LightGBM ship separate OpenMP runtimes on this workstation.  This
runner deliberately imports no LightGBM code, preventing the silent runtime
conflict that can occur when both frameworks execute in one Python process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optiver.blending import align_oof_frames  # noqa: E402
from optiver.data import load_feature_table  # noqa: E402
from optiver.mlp import CURATED_MLP_FEATURES, MLPConfig, run_mlp_cv  # noqa: E402
from optiver.nested_blending import hash_group_ids, sha256_file  # noqa: E402
from optiver.validation import build_group_folds  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, required=True)
    parser.add_argument("--lightgbm-oof", type=Path, required=True)
    parser.add_argument("--mlp-oof", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
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
        PROJECT_ROOT / "src" / "optiver" / "blending.py",
        PROJECT_ROOT / "src" / "optiver" / "data.py",
        PROJECT_ROOT / "src" / "optiver" / "metrics.py",
        PROJECT_ROOT / "src" / "optiver" / "mlp.py",
        PROJECT_ROOT / "src" / "optiver" / "nested_blending.py",
        PROJECT_ROOT / "src" / "optiver" / "validation.py",
    ]
    return {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in sources
        if path.is_file()
    }


def _load_outer_oof(lightgbm_path: Path, mlp_path: Path):
    columns = ["stock_id", "time_id", "target", "fold", "prediction"]
    aligned = align_oof_frames(
        {
            "lightgbm": pd.read_parquet(lightgbm_path, columns=columns),
            "mlp": pd.read_parquet(mlp_path, columns=columns),
        }
    )
    if aligned.model_names != ("lightgbm", "mlp"):
        raise RuntimeError(f"Unexpected model order: {aligned.model_names!r}")
    if sorted(int(x) for x in np.unique(aligned.fold)) != list(range(5)):
        raise ValueError("Outer OOF must contain folds 0..4")
    return aligned


def _assert_frame_matches(frame: pd.DataFrame, aligned: Any) -> None:
    reference = aligned.keys.copy()
    reference["target"] = aligned.target
    observed = frame[["stock_id", "time_id", "target"]]
    merged = reference.merge(
        observed,
        on=["stock_id", "time_id"],
        suffixes=("_oof", "_features"),
        validate="one_to_one",
    )
    if len(merged) != len(reference) or len(observed) != len(reference):
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


def _execute(args: argparse.Namespace, output_dir: Path) -> None:
    feature_path = args.features_path.expanduser().resolve()
    lightgbm_oof_path = _resolve_oof(args.lightgbm_oof)
    mlp_oof_path = _resolve_oof(args.mlp_oof)
    aligned = _load_outer_oof(lightgbm_oof_path, mlp_oof_path)

    schema = pq.ParquetFile(feature_path).schema_arrow.names
    missing = [name for name in CURATED_MLP_FEATURES if name not in schema]
    if missing:
        raise ValueError(f"Curated MLP features are missing: {missing}")
    needed = ["stock_id", "time_id", "target", *CURATED_MLP_FEATURES]
    frame = load_feature_table(feature_path, columns=needed)
    _assert_frame_matches(frame, aligned)

    group_to_fold = _group_fold_map(aligned)
    row_outer_fold = frame["time_id"].map(group_to_fold)
    if row_outer_fold.isna().any():
        raise ValueError("Feature table contains groups absent from outer OOF")
    outer_validation_mask = row_outer_fold.to_numpy(dtype=np.int16) == args.outer_fold
    outer_train = frame.loc[~outer_validation_mask].reset_index(drop=True)
    calibration_seed = args.seed + 40_000 + args.outer_fold
    split = build_group_folds(
        outer_train[["time_id"]],
        group_col="time_id",
        n_splits=args.inner_splits,
        strategy="group_kfold",
        random_state=calibration_seed,
    )[0]
    fit_idx, calibration_idx = split.train_idx, split.valid_idx
    fit_groups = tuple(
        sorted(int(x) for x in pd.unique(outer_train.iloc[fit_idx]["time_id"]))
    )
    calibration_groups = tuple(
        sorted(
            int(x)
            for x in pd.unique(outer_train.iloc[calibration_idx]["time_id"])
        )
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
    config = MLPConfig(
        hidden_dims=(256, 128, 64),
        embedding_dim=16,
        dropout=0.20,
        batch_size=8_192,
        learning_rate=3e-4,
        weight_decay=1e-4,
        max_epochs=80,
        patience=10,
        min_delta=1e-5,
        fixed_epochs=None,
        inner_splits=args.inner_splits,
        gradient_clip_norm=1.0,
        feature_clip_value=12.0,
        seed=model_seed,
        device=args.device,
        num_workers=0,
    )
    result = run_mlp_cv(
        outer_train,
        [(fit_idx, calibration_idx)],
        CURATED_MLP_FEATURES,
        config=config,
        model_dir=output_dir / "models",
        allow_partial_oof=True,
    )
    prediction = result.oof_predictions[calibration_idx]
    prediction_frame = outer_train.iloc[calibration_idx][
        ["stock_id", "time_id", "target"]
    ].reset_index(drop=True)
    prediction_frame["prediction"] = prediction
    prediction_path = output_dir / "mlp_calibration_predictions.parquet"
    prediction_frame.to_parquet(prediction_path, index=False)
    result.fold_metrics.to_csv(output_dir / "mlp_fold_metrics.csv", index=False)
    result.training_history.to_csv(
        output_dir / "mlp_training_history.csv", index=False
    )
    _json_dump(output_dir / "mlp_summary.json", result.summary)

    model_config = asdict(config) | {
        "feature_names": list(CURATED_MLP_FEATURES)
    }
    source_hashes = _source_hashes()
    manifest = {
        "schema_version": "nested_calibration_component_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "mlp",
        "outer_fold": args.outer_fold,
        "group_column": "time_id",
        "fit_groups": list(fit_groups),
        "calibration_groups": list(calibration_groups),
        "outer_train_groups": list(outer_train_groups),
        "outer_validation_groups": list(outer_validation_groups),
        "group_hashes": {
            "fit_groups_sha256": hash_group_ids(fit_groups),
            "calibration_groups_sha256": hash_group_ids(calibration_groups),
            "outer_train_groups_sha256": hash_group_ids(outer_train_groups),
            "outer_validation_groups_sha256": hash_group_ids(
                outer_validation_groups
            ),
        },
        "training_scope": "fit_groups_only",
        "training_seed": model_seed,
        "calibration_seed": calibration_seed,
        "model_config": model_config,
        "model_config_sha256": _canonical_sha256(model_config),
        "source_files_sha256": source_hashes,
        "outer_prediction_file": str(mlp_oof_path),
        "outer_prediction_file_sha256": sha256_file(mlp_oof_path),
        "predictions": {
            "path": prediction_path.name,
            "sha256": sha256_file(prediction_path),
            "rows": int(len(prediction_frame)),
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
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
    }
    _json_dump(output_dir / "component_manifest.json", manifest)


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
            "component_manifest_sha256": sha256_file(
                output_dir / "component_manifest.json"
            ),
        }
        _json_dump(output_dir / "COMPLETED.json", completion)
        _json_dump(status_path, completion)


if __name__ == "__main__":
    main()
