#!/usr/bin/env python3
"""Run a reproducible, leakage-safe grouped-CV MLP experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optiver.data import load_feature_table  # noqa: E402
from optiver.feature_sets import FEATURE_SET_GROUPS, get_feature_set  # noqa: E402
from optiver.mlp import (  # noqa: E402
    CURATED_MLP_FEATURES,
    MLPConfig,
    resolve_device,
    run_mlp_cv,
)
from optiver.validation import GroupFold, build_group_folds  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "artifacts"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--feature-set",
        choices=["curated_mlp", "rv_only", *FEATURE_SET_GROUPS],
        default="curated_mlp",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--max-folds",
        type=int,
        help="Pilot only: train the first N outer folds and leave other OOF rows empty",
    )
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--split-strategy",
        choices=["group_kfold", "feature_balanced"],
        default="group_kfold",
    )
    parser.add_argument(
        "--fixed-epochs",
        type=int,
        help="Use this pre-registered epoch count instead of inner early stopping",
    )
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument(
        "--hidden-dims",
        default="256,128,64",
        help="Comma-separated hidden layer widths",
    )
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--feature-clip-value", type=float, default=12.0)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _parse_hidden_dims(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("--hidden-dims must contain comma-separated integers") from exc
    if not result or any(width < 1 for width in result):
        raise ValueError("--hidden-dims must contain positive layer widths")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _source_file_hashes() -> dict[str, str]:
    """Hash the executable source needed to reproduce an MLP run."""

    sources = [
        PROJECT_ROOT / "pyproject.toml",
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "optiver" / "data.py",
        PROJECT_ROOT / "src" / "optiver" / "metrics.py",
        PROJECT_ROOT / "src" / "optiver" / "mlp.py",
        PROJECT_ROOT / "src" / "optiver" / "validation.py",
    ]
    return {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path)
        for path in sources
        if path.is_file()
    }


def _fold_pairs(folds: list[GroupFold]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(fold.train_idx, fold.valid_idx) for fold in folds]


def _fold_assignment(frame: pd.DataFrame, folds: list[GroupFold]) -> pd.DataFrame:
    assignment = np.full(len(frame), -1, dtype=np.int16)
    for fold in folds:
        assignment[fold.valid_idx] = fold.fold_id
    result = frame[["time_id"]].copy()
    result["fold"] = assignment
    per_time = result.drop_duplicates()
    if per_time["time_id"].duplicated().any() or (per_time["fold"] < 0).any():
        raise RuntimeError("Fold map is not one-to-one by time_id")
    return per_time.sort_values("time_id").reset_index(drop=True)


def _make_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    seed: int,
    strategy: str,
) -> list[GroupFold]:
    kwargs: dict[str, Any] = {}
    if strategy == "feature_balanced":
        kwargs.update(feature_cols=["rv_pred"], entity_col="stock_id")
    return build_group_folds(
        frame,
        group_col="time_id",
        n_splits=n_splits,
        strategy=strategy,
        random_state=seed,
        **kwargs,
    )


def _environment_versions() -> dict[str, str]:
    import pyarrow
    import torch

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "torch": torch.__version__,
    }


def _execute_run(
    args: argparse.Namespace,
    *,
    feature_path: Path,
    output_dir: Path,
) -> None:
    import pyarrow.parquet as pq

    schema = pq.ParquetFile(feature_path).schema_arrow.names
    if args.feature_set == "curated_mlp":
        missing = [name for name in CURATED_MLP_FEATURES if name not in schema]
        if missing:
            raise ValueError(f"Curated MLP feature set is missing columns: {missing}")
        feature_names = list(CURATED_MLP_FEATURES)
    else:
        feature_names = get_feature_set(args.feature_set, schema, strict=True)
    needed = list(
        dict.fromkeys(["stock_id", "time_id", "target", *feature_names])
    )
    frame = load_feature_table(feature_path, columns=needed)
    folds = _make_folds(
        frame,
        n_splits=args.n_splits,
        seed=args.seed,
        strategy=args.split_strategy,
    )
    if args.max_folds is not None:
        if not 1 <= args.max_folds <= len(folds):
            raise ValueError(
                f"--max-folds must be between 1 and {len(folds)} inclusive"
            )
        run_folds = folds[: args.max_folds]
    else:
        run_folds = folds
    fold_map_path = output_dir / "fold_assignments.csv"
    fold_map = _fold_assignment(frame, folds)
    fold_map.to_csv(fold_map_path, index=False)

    config = MLPConfig(
        hidden_dims=_parse_hidden_dims(args.hidden_dims),
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        fixed_epochs=args.fixed_epochs,
        inner_splits=args.inner_splits,
        gradient_clip_norm=args.gradient_clip_norm,
        feature_clip_value=args.feature_clip_value,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
    )
    resolved_device = resolve_device(config.device)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": vars(args)
        | {
            "features_path": str(args.features_path),
            "output_root": str(args.output_root),
        },
        "config": asdict(config),
        "data": {
            "path": str(feature_path),
            "size_bytes": feature_path.stat().st_size,
            "sha256": _sha256(feature_path),
            "rows": len(frame),
            "stocks": int(frame["stock_id"].nunique()),
            "time_ids": int(frame["time_id"].nunique()),
        },
        "environment": _environment_versions(),
        "source_files_sha256": _source_file_hashes(),
        "fold_map": {
            "path": fold_map_path.name,
            "sha256": _sha256(fold_map_path),
            "rows": int(len(fold_map)),
        },
        "effective_parameters": {
            "experiment": "mlp",
            "feature_set": args.feature_set,
            "feature_names": feature_names,
            "validation": {
                "outer_n_splits": args.n_splits,
                "inner_n_splits": args.inner_splits,
                "split_strategy": args.split_strategy,
                "group": "time_id",
                "seed": args.seed,
                "max_folds": args.max_folds,
            },
            "training": asdict(config),
        },
        "reproducibility": {
            "requested_device": config.device,
            "resolved_device": resolved_device,
            "base_seed": config.seed,
            "outer_fold_seed_rule": "base_seed + fold_id",
            "inner_model_seed_rule": "base_seed + 10000 + fold_id",
            "deterministic_algorithms": True,
            "pilot_max_folds": args.max_folds,
        },
    }
    _json_dump(output_dir / "run_metadata.json", metadata)

    result = run_mlp_cv(
        frame,
        _fold_pairs(run_folds),
        feature_names,
        config=config,
        model_dir=output_dir / "models",
        allow_partial_oof=args.max_folds is not None,
    )
    oof = frame[["stock_id", "time_id", "target"]].copy()
    oof["fold"] = result.fold_ids
    oof["prediction"] = result.oof_predictions
    oof["is_oof"] = result.fold_ids >= 0
    oof.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    result.fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    result.training_history.to_csv(output_dir / "training_history.csv", index=False)
    _json_dump(output_dir / "summary.json", result.summary)
    _json_dump(output_dir / "feature_manifest.json", result.feature_names)
    print(json.dumps(result.summary, indent=2, sort_keys=True))


def main() -> None:
    args = _parse_args()
    feature_path = args.features_path.expanduser().resolve()
    output_dir = args.output_root.expanduser().resolve() / args.run_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty run directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    status_path = output_dir / "status.json"
    _json_dump(status_path, {"status": "running", "started_at_utc": started_at})
    try:
        _execute_run(args, feature_path=feature_path, output_dir=output_dir)
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
            "run_metadata_sha256": _sha256(output_dir / "run_metadata.json"),
        }
        _json_dump(output_dir / "COMPLETED.json", completion)
        _json_dump(status_path, completion)


if __name__ == "__main__":
    main()
