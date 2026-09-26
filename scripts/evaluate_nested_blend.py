#!/usr/bin/env python3
"""Evaluate a strictly nested blend from outer OOF and calibration bundles.

Each ``--calibration-bundle`` must contain ``calibration_manifest.json`` and
the long-form prediction parquet named by that manifest.  One bundle is
required for every outer fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optiver.blending import align_oof_frames  # noqa: E402
from optiver.nested_blending import (  # noqa: E402
    load_nested_calibration_bundle,
    nested_calibrated_blend,
)


def _parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label or not raw_path.strip():
        raise argparse.ArgumentTypeError("--model label and path must not be empty")
    return label, Path(raw_path).expanduser()


def _resolve_oof_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_dir():
        resolved = resolved / "oof_predictions.parquet"
    if not resolved.is_file():
        raise FileNotFoundError(f"OOF parquet does not exist: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _source_file_hashes() -> dict[str, str]:
    sources = [
        PROJECT_ROOT / "pyproject.toml",
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "optiver" / "blending.py",
        PROJECT_ROOT / "src" / "optiver" / "nested_blending.py",
        PROJECT_ROOT / "src" / "optiver" / "metrics.py",
    ]
    return {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path)
        for path in sources
        if path.is_file()
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        type=_parse_model,
        required=True,
        help="Outer OOF run as LABEL=PARQUET_OR_RUN_DIRECTORY; repeat 2+ times",
    )
    parser.add_argument(
        "--calibration-bundle",
        action="append",
        type=Path,
        required=True,
        help="Fold bundle directory or calibration_manifest.json; repeat per fold",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


def _execute_run(args: argparse.Namespace, *, output_dir: Path) -> None:
    if len(args.model) < 2:
        raise ValueError("Pass at least two --model LABEL=PATH arguments")
    labels = [label for label, _ in args.model]
    if len(set(labels)) != len(labels):
        raise ValueError("Every --model label must be unique")

    frames: dict[str, pd.DataFrame] = {}
    outer_input_records: list[dict[str, object]] = []
    outer_hash_by_model: dict[str, str] = {}
    for label, path in args.model:
        oof_path = _resolve_oof_path(path)
        file_hash = _sha256(oof_path)
        outer_hash_by_model[label] = file_hash
        frames[label] = pd.read_parquet(
            oof_path,
            columns=["stock_id", "time_id", "target", "fold", "prediction"],
        )
        outer_input_records.append(
            {
                "model": label,
                "path": str(oof_path),
                "size_bytes": oof_path.stat().st_size,
                "sha256": file_hash,
            }
        )
    outer_oof = align_oof_frames(frames)

    calibration_folds = {}
    calibration_records: list[dict[str, object]] = []
    for supplied_path in args.calibration_bundle:
        calibration = load_nested_calibration_bundle(
            supplied_path, outer_oof=outer_oof
        )
        if calibration.outer_fold in calibration_folds:
            raise ValueError(
                f"Duplicate calibration bundle for fold {calibration.outer_fold}"
            )
        # When generators bind a bundle to exact outer OOF artifacts, enforce
        # those optional hashes. The group provenance remains mandatory.
        for model_name, model_record in calibration.source_manifest["models"].items():
            claimed_outer_hash = model_record.get("outer_prediction_file_sha256")
            if claimed_outer_hash is not None and (
                claimed_outer_hash != outer_hash_by_model[model_name]
            ):
                raise ValueError(
                    f"Outer OOF file hash mismatch for {model_name}, "
                    f"fold {calibration.outer_fold}"
                )
        calibration_folds[calibration.outer_fold] = calibration
        manifest_path = (
            supplied_path.expanduser().resolve() / "calibration_manifest.json"
            if supplied_path.expanduser().resolve().is_dir()
            else supplied_path.expanduser().resolve()
        )
        calibration_records.append(
            {
                "outer_fold": calibration.outer_fold,
                "manifest_path": str(manifest_path),
                "manifest_file_sha256": _sha256(manifest_path),
                "manifest_canonical_sha256": calibration.manifest_sha256,
                "prediction_file_sha256": calibration.prediction_file_sha256,
            }
        )

    result = nested_calibrated_blend(
        outer_oof,
        calibration_folds,
        bootstrap_resamples=args.bootstrap_resamples,
        confidence=args.confidence,
        seed=args.seed,
    )
    oof = outer_oof.keys.copy()
    oof["target"] = outer_oof.target
    oof["fold"] = outer_oof.fold
    oof["prediction"] = result.oof_predictions
    for column, model_name in enumerate(outer_oof.model_names):
        oof[f"prediction__{model_name}"] = outer_oof.predictions[:, column]
    oof.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    result.fold_weights.to_csv(output_dir / "fold_weights.csv", index=False)
    result.fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    result.model_summary.to_csv(output_dir / "model_summary.csv", index=False)
    result.paired_comparisons.to_csv(
        output_dir / "paired_comparisons.csv", index=False
    )
    _json_dump(output_dir / "summary.json", result.summary)
    _json_dump(
        output_dir / "validated_calibration_provenance.json",
        {
            str(fold): calibration.source_manifest
            for fold, calibration in sorted(calibration_folds.items())
        },
    )

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": {
            "models": [f"{label}={path}" for label, path in args.model],
            "calibration_bundles": [str(path) for path in args.calibration_bundle],
            "output_dir": str(args.output_dir),
            "bootstrap_resamples": args.bootstrap_resamples,
            "confidence": args.confidence,
            "seed": args.seed,
        },
        "outer_oof_inputs": outer_input_records,
        "calibration_inputs": calibration_records,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "source_files_sha256": _source_file_hashes(),
        "method": {
            "name": "strict_nested_calibration_blend",
            "validation_claim": "strict_nested_outer_oof",
            "objective": "RMSPE (weighted least squares in relative-error space)",
            "constraint": "component weights >= 0 and sum to 1",
            "weight_fit": (
                "For outer fold k, fit weights only on k's inner-calibration "
                "predictions. Those predictions are generated by component "
                "models trained on the disjoint inner-fit groups inside outer-train."
            ),
            "provenance_validation": (
                "Complete group IDs and SHA-256 hashes are checked against the "
                "outer OOF fold map and calibration parquet before evaluation."
            ),
            "prediction_clipping": "component predictions clipped at zero before fitting",
            "uncertainty": "paired bootstrap resampling complete time_id groups",
            "uncertainty_caveat": (
                "Intervals condition on the fitted base models and fold-local "
                "calibration weights; they do not include model-training or "
                "calibration-selection uncertainty."
            ),
        },
    }
    _json_dump(output_dir / "run_metadata.json", metadata)

    print(json.dumps(result.summary, indent=2, sort_keys=True))
    print("\nStrict nested fold weights:")
    print(result.fold_weights.to_string(index=False))
    print("\nPaired differences (negative favors strict nested blend):")
    print(result.paired_comparisons.to_string(index=False))


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
        _execute_run(args, output_dir=output_dir)
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
