#!/usr/bin/env python3
"""Verify a local Optiver feature cache against the committed manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "feature_cache_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_path", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    feature_path = args.feature_path.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    if not feature_path.is_file():
        raise FileNotFoundError(f"Feature cache not found: {feature_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    parquet = pq.ParquetFile(feature_path)
    columns = set(parquet.schema_arrow.names)
    required = {*manifest["key_columns"], manifest["target_column"]}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"Feature cache is missing required columns: {missing}")

    keys = pd.read_parquet(
        feature_path,
        columns=[*manifest["key_columns"], manifest["target_column"]],
    )
    observed = {
        "sha256": _sha256(feature_path),
        "size_bytes": feature_path.stat().st_size,
        "rows": len(keys),
        "stocks": int(keys["stock_id"].nunique()),
        "time_ids": int(keys["time_id"].nunique()),
    }
    mismatches = {
        name: {"expected": manifest[name], "observed": value}
        for name, value in observed.items()
        if value != manifest[name]
    }
    if keys.duplicated(manifest["key_columns"]).any():
        mismatches["duplicate_keys"] = True
    if keys[manifest["target_column"]].isna().any():
        mismatches["missing_targets"] = True
    if mismatches:
        raise ValueError(
            "Feature cache does not match the canonical manifest:\n"
            + json.dumps(mismatches, indent=2, sort_keys=True)
        )
    print(json.dumps({"status": "ok", **observed}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
