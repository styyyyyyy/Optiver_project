#!/usr/bin/env python3
"""Run descriptive post-hoc blending of two or more OOF runs.

This command is retained for exploratory analysis only.  It is not a strict
nested-CV estimate because other-fold OOF predictions can indirectly depend on
the nominally held-out fold.  Use ``evaluate_nested_blend.py`` for reporting.

Example
-------
python scripts/evaluate_blend.py \
  --model lightgbm=artifacts/c1i1_lightgbm \
  --model mlp=artifacts/clean_mlp \
  --output-dir artifacts/lightgbm_mlp_blend
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

from optiver.blending import align_oof_frames, posthoc_oof_blend  # noqa: E402


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
        help="Component run as LABEL=OOF_PARQUET_OR_RUN_DIRECTORY; repeat 2+ times",
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
    input_records: list[dict[str, object]] = []
    for label, path in args.model:
        oof_path = _resolve_oof_path(path)
        frames[label] = pd.read_parquet(
            oof_path,
            columns=["stock_id", "time_id", "target", "fold", "prediction"],
        )
        input_records.append(
            {
                "model": label,
                "path": str(oof_path),
                "size_bytes": oof_path.stat().st_size,
                "sha256": _sha256(oof_path),
            }
        )

    aligned = align_oof_frames(frames)
    result = posthoc_oof_blend(
        aligned,
        bootstrap_resamples=args.bootstrap_resamples,
        confidence=args.confidence,
        seed=args.seed,
    )

    oof = aligned.keys.copy()
    oof["target"] = aligned.target
    oof["fold"] = aligned.fold
    oof["prediction"] = result.oof_predictions
    for column, model_name in enumerate(aligned.model_names):
        oof[f"prediction__{model_name}"] = aligned.predictions[:, column]
    oof.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    result.fold_weights.to_csv(output_dir / "fold_weights.csv", index=False)
    result.deployment_weights.to_csv(
        output_dir / "deployment_weights.csv", index=False
    )
    result.fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    result.model_summary.to_csv(output_dir / "model_summary.csv", index=False)
    result.paired_comparisons.to_csv(
        output_dir / "paired_comparisons.csv", index=False
    )
    _json_dump(output_dir / "summary.json", result.summary)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": {
            "models": [f"{label}={path}" for label, path in args.model],
            "output_dir": str(args.output_dir),
            "bootstrap_resamples": args.bootstrap_resamples,
            "confidence": args.confidence,
            "seed": args.seed,
        },
        "inputs": input_records,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "source_files_sha256": _source_file_hashes(),
        "method": {
            "name": "posthoc_oof_blend",
            "validation_claim": "descriptive_posthoc_not_strict_nested",
            "indirect_dependency_warning": (
                "Other-fold OOF predictions may be produced by base models "
                "trained on the nominally held-out fold. Do not report this "
                "score as an unbiased nested-CV estimate."
            ),
            "objective": "RMSPE (weighted least squares in relative-error space)",
            "constraint": "component weights >= 0 and sum to 1",
            "weight_fit": (
                "For held-out outer fold k, fit weights only on OOF rows whose "
                "outer fold is not k; apply those weights to k. This removes "
                "direct target reuse but not indirect base-model dependence."
            ),
            "two_model_solver": "closed-form constrained optimum",
            "multi_model_solver": "deterministic projected gradient on simplex",
            "prediction_clipping": "component OOF predictions clipped at zero before fitting",
            "uncertainty": "paired bootstrap resampling complete time_id groups",
        },
    }
    _json_dump(output_dir / "run_metadata.json", metadata)

    print(json.dumps(result.summary, indent=2, sort_keys=True))
    print("\nFold-specific weights:")
    print(result.fold_weights.to_string(index=False))
    print("\nPaired differences (negative favors post-hoc blend):")
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
