"""Feature-table loading and schema checks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


KEY_COLUMNS = ("stock_id", "time_id")
TARGET_COLUMN = "target"
_LEGACY_CLUSTER_PATTERN = re.compile(r"_\d+c1$")


def target_derived_columns(columns: Iterable[str]) -> list[str]:
    """Identify columns contaminated by the legacy full-target clustering."""

    return sorted(
        column
        for column in columns
        if column == "stock_cluster" or _LEGACY_CLUSTER_PATTERN.search(column)
    )


def observable_feature_columns(columns: Iterable[str]) -> list[str]:
    """Return model features available without labels at inference time."""

    excluded = set(KEY_COLUMNS) | {TARGET_COLUMN} | set(target_derived_columns(columns))
    return [column for column in columns if column not in excluded]


def validate_feature_table(frame: pd.DataFrame, *, require_target: bool = True) -> None:
    required = set(KEY_COLUMNS)
    if require_target:
        required.add(TARGET_COLUMN)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Feature table is missing required columns: {missing}")

    if frame.loc[:, list(KEY_COLUMNS)].isna().any().any():
        raise ValueError("stock_id/time_id keys contain missing values")
    duplicate_count = int(frame.duplicated(list(KEY_COLUMNS)).sum())
    if duplicate_count:
        raise ValueError(
            f"Expected one row per (stock_id, time_id); found {duplicate_count} duplicates"
        )
    if require_target:
        target = frame[TARGET_COLUMN].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(target)) or np.any(target <= 0):
            raise ValueError("target must be finite and strictly positive")


def load_feature_table(
    path: str | Path,
    *,
    columns: list[str] | None = None,
    require_target: bool = True,
) -> pd.DataFrame:
    """Load a parquet feature table and fail fast on schema problems."""

    feature_path = Path(path).expanduser()
    if not feature_path.is_file():
        raise FileNotFoundError(f"Feature table not found: {feature_path}")
    frame = pd.read_parquet(feature_path, columns=columns)
    validate_feature_table(frame, require_target=require_target)
    return frame
