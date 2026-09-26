"""Leakage-safe validation helpers for the Optiver experiments.

The competition's ``time_id`` is an anonymised group identifier, not a
timestamp.  This module therefore never sorts it to create a pseudo-time
series.  The primary protocol is grouped cross-validation: all rows sharing a
``time_id`` are held out together.

``feature_balanced`` folds are provided only as a robustness protocol.  They
spread groups through feature space in the spirit of the original project,
but callers must pass *observable, non-target* features.  Model selection and
the headline score should use ``group_kfold``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


FoldStrategy = Literal["group_kfold", "feature_balanced"]


@dataclass(frozen=True)
class GroupFold:
    """One grouped train/validation split.

    ``train_idx`` and ``valid_idx`` are zero-based *positional* row indices,
    suitable for ``DataFrame.iloc``.  They deliberately do not refer to the
    input frame's index labels, which may be non-unique.
    """

    fold_id: int
    train_idx: np.ndarray
    valid_idx: np.ndarray
    train_groups: tuple[object, ...]
    valid_groups: tuple[object, ...]


@dataclass(frozen=True)
class CrossFittedTargetStats:
    """Stock target statistics for one outer fold.

    ``train`` contains inner out-of-fold encodings for the outer-training
    rows.  ``validation`` is fitted once on all outer-training rows and then
    applied to the outer-validation rows.  The validation target, if present,
    is never read.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    feature_names: tuple[str, ...]
    inner_folds: tuple[GroupFold, ...]

    @property
    def valid(self) -> pd.DataFrame:
        """Short alias for ``validation``."""

        return self.validation


def build_group_folds(
    data: pd.DataFrame,
    *,
    group_col: str = "time_id",
    n_splits: int = 5,
    strategy: FoldStrategy = "group_kfold",
    feature_cols: Sequence[str] | None = None,
    entity_col: str | None = None,
    random_state: int = 2021,
    target_col: str | None = "target",
) -> list[GroupFold]:
    """Build reproducible folds that keep each group wholly in one fold.

    Parameters
    ----------
    data:
        Row-level modelling data.
    group_col:
        Group held out as a unit.  For this project it should be ``time_id``.
    n_splits:
        Number of outer or inner folds.  There must be at least this many
        distinct groups.
    strategy:
        ``"group_kfold"`` is the primary, target-agnostic protocol.
        ``"feature_balanced"`` is an optional robustness protocol that tries
        to make every validation fold diverse in observable feature space.
    feature_cols:
        Required only for ``feature_balanced``.  Never pass the prediction
        target (the default target name is explicitly rejected).
    entity_col:
        If supplied for ``feature_balanced``, each feature is pivoted to a
        group-by-entity matrix.  With ``group_col="time_id"``,
        ``entity_col="stock_id"`` and ``feature_cols=["rv_pred"]`` this is the
        leakage-safe analogue of the project's previous feature-space folds.
        Without it, features are averaged within each group.
    random_state:
        Local seed used to freeze the group-to-fold mapping for both
        strategies.  Global NumPy random state is never modified.
    target_col:
        Column name to reject from ``feature_cols``.  Set to ``None`` only
        when the caller has independently established that no target-derived
        feature is used.

    Returns
    -------
    list[GroupFold]
        Folds with positional row indices.  Coverage and disjointness are
        checked before the list is returned.
    """

    _require_columns(data, [group_col])
    _validate_group_values(data[group_col], group_col)
    _validate_n_splits(n_splits, data[group_col].nunique(dropna=False))

    if strategy == "group_kfold":
        groups = data[group_col].to_numpy(copy=False)
        dummy = np.zeros((len(data), 1), dtype=np.uint8)
        split_indices = _seeded_group_kfold_indices(
            dummy,
            groups=groups,
            n_splits=n_splits,
            random_state=random_state,
        )
        folds = [
            _make_fold(data, group_col, fold_id, train_idx, valid_idx)
            for fold_id, (train_idx, valid_idx) in enumerate(
                split_indices
            )
        ]
    elif strategy == "feature_balanced":
        if not feature_cols:
            raise ValueError(
                "feature_cols must contain at least one observable feature "
                "when strategy='feature_balanced'"
            )
        feature_cols = tuple(feature_cols)
        _require_columns(data, [*feature_cols, *([entity_col] if entity_col else [])])
        if target_col is not None and target_col in feature_cols:
            raise ValueError(
                f"Target column {target_col!r} cannot be used to balance folds"
            )

        group_values, matrix = _group_feature_matrix(
            data,
            group_col=group_col,
            feature_cols=feature_cols,
            entity_col=entity_col,
        )
        assignments = _balanced_assignments(
            matrix,
            n_splits=n_splits,
            random_state=random_state,
        )
        folds = []
        for fold_id, group_positions in enumerate(assignments):
            valid_groups = group_values[np.asarray(group_positions, dtype=int)]
            valid_mask = data[group_col].isin(valid_groups).to_numpy()
            valid_idx = np.flatnonzero(valid_mask)
            train_idx = np.flatnonzero(~valid_mask)
            folds.append(
                _make_fold(data, group_col, fold_id, train_idx, valid_idx)
            )
    else:
        raise ValueError(
            "strategy must be either 'group_kfold' or 'feature_balanced'; "
            f"got {strategy!r}"
        )

    assert_valid_group_folds(folds, data, group_col=group_col)
    return folds


def assert_valid_group_folds(
    folds: Sequence[GroupFold],
    data: pd.DataFrame,
    *,
    group_col: str = "time_id",
) -> None:
    """Assert row coverage, group coverage, and train/validation isolation.

    A valid fold collection has these invariants:

    * every row occurs in validation exactly once;
    * every group occurs in validation in exactly one fold;
    * a fold's train and validation rows are exact complements; and
    * no group appears on both sides of any fold.

    ``AssertionError`` is used intentionally so this function can serve as a
    pre-training guard in scripts and in tests.
    """

    _require_columns(data, [group_col])
    if not folds:
        raise AssertionError("At least one fold is required")

    n_rows = len(data)
    all_rows = np.arange(n_rows, dtype=int)
    validation_counts = np.zeros(n_rows, dtype=np.int64)
    group_owner: dict[object, int] = {}
    fold_ids: set[int] = set()

    for fold in folds:
        if fold.fold_id in fold_ids:
            raise AssertionError(f"Duplicate fold_id: {fold.fold_id}")
        fold_ids.add(fold.fold_id)

        train_idx = _checked_indices(fold.train_idx, n_rows, "train_idx")
        valid_idx = _checked_indices(fold.valid_idx, n_rows, "valid_idx")
        if train_idx.size == 0 or valid_idx.size == 0:
            raise AssertionError(f"Fold {fold.fold_id} has an empty split")
        if np.intersect1d(train_idx, valid_idx).size:
            raise AssertionError(
                f"Fold {fold.fold_id} has rows in both train and validation"
            )
        if not np.array_equal(
            np.sort(np.concatenate([train_idx, valid_idx])), all_rows
        ):
            raise AssertionError(
                f"Fold {fold.fold_id} train/validation rows are not complements"
            )

        train_groups = set(data.iloc[train_idx][group_col].tolist())
        valid_groups = set(data.iloc[valid_idx][group_col].tolist())
        overlap = train_groups.intersection(valid_groups)
        if overlap:
            raise AssertionError(
                f"Fold {fold.fold_id} leaks groups across train/validation: "
                f"{_short_values(overlap)}"
            )

        if train_groups != set(fold.train_groups):
            raise AssertionError(f"Fold {fold.fold_id} train_groups metadata is stale")
        if valid_groups != set(fold.valid_groups):
            raise AssertionError(f"Fold {fold.fold_id} valid_groups metadata is stale")

        for group in valid_groups:
            previous_owner = group_owner.get(group)
            if previous_owner is not None:
                raise AssertionError(
                    f"Group {group!r} is validated in folds "
                    f"{previous_owner} and {fold.fold_id}"
                )
            group_owner[group] = fold.fold_id

        validation_counts[valid_idx] += 1

    bad_rows = np.flatnonzero(validation_counts != 1)
    if bad_rows.size:
        raise AssertionError(
            "Validation rows must have exactly-once coverage; bad positions: "
            f"{bad_rows[:10].tolist()}"
        )

    expected_groups = set(data[group_col].tolist())
    if set(group_owner) != expected_groups:
        missing = expected_groups.difference(group_owner)
        raise AssertionError(
            "Validation folds do not cover every group; missing: "
            f"{_short_values(missing)}"
        )


def cross_fit_stock_target_stats(
    outer_train: pd.DataFrame,
    outer_valid: pd.DataFrame,
    *,
    stock_col: str = "stock_id",
    target_col: str = "target",
    group_col: str = "time_id",
    inner_splits: int = 5,
    random_state: int = 2021,
    prefix: str = "stock_target",
    include_count: bool = False,
) -> CrossFittedTargetStats:
    """Create leakage-safe stock target statistics inside one outer fold.

    The outer-training rows are encoded with inner ``GroupKFold``.  Thus a
    row's statistics use targets only from *other* ``time_id`` groups, without
    inventing a chronological interpretation for the anonymised identifier.
    Outer-validation rows are encoded from all outer-training targets.

    Unknown stocks fall back to the relevant training partition's global
    mean, median, and population standard deviation.  Their optional count is
    zero.  The input frames are not modified and their original indices are
    preserved in the returned feature frames.
    """

    if not prefix or not isinstance(prefix, str):
        raise ValueError("prefix must be a non-empty string")
    _require_columns(outer_train, [stock_col, target_col, group_col])
    _require_columns(outer_valid, [stock_col, group_col])
    _validate_group_values(outer_train[group_col], group_col)
    _validate_group_values(outer_valid[group_col], group_col)
    _validate_non_null(outer_train[stock_col], stock_col)
    _validate_non_null(outer_valid[stock_col], stock_col)

    outer_train_groups = set(outer_train[group_col].tolist())
    outer_valid_groups = set(outer_valid[group_col].tolist())
    overlap = outer_train_groups.intersection(outer_valid_groups)
    if overlap:
        raise ValueError(
            "outer_train and outer_valid must have disjoint group values; "
            f"overlap: {_short_values(overlap)}"
        )

    targets = _finite_target_values(outer_train[target_col], target_col)
    # Work with an explicitly numeric target copy.  This also ensures that a
    # numeric-looking object/string column is treated consistently.
    train_numeric = outer_train.copy()
    train_numeric[target_col] = targets

    inner_folds = build_group_folds(
        train_numeric,
        group_col=group_col,
        n_splits=inner_splits,
        strategy="group_kfold",
        random_state=random_state,
    )

    suffixes = ["mean", "median", "std"]
    if include_count:
        suffixes.append("count")
    feature_names = tuple(f"{prefix}_{suffix}" for suffix in suffixes)

    train_values = np.full(
        (len(train_numeric), len(feature_names)), np.nan, dtype=float
    )
    for fold in inner_folds:
        fitted = _fit_stock_statistics(
            train_numeric.iloc[fold.train_idx],
            stock_col=stock_col,
            target_col=target_col,
            prefix=prefix,
        )
        transformed = _apply_stock_statistics(
            train_numeric.iloc[fold.valid_idx],
            fitted,
            stock_col=stock_col,
            feature_names=feature_names,
        )
        train_values[fold.valid_idx] = transformed.to_numpy(dtype=float)

    if not np.isfinite(train_values).all():
        raise RuntimeError("Cross-fitted training statistics contain non-finite values")

    outer_fitted = _fit_stock_statistics(
        train_numeric,
        stock_col=stock_col,
        target_col=target_col,
        prefix=prefix,
    )
    validation_features = _apply_stock_statistics(
        outer_valid,
        outer_fitted,
        stock_col=stock_col,
        feature_names=feature_names,
    )
    train_features = pd.DataFrame(
        train_values,
        columns=feature_names,
        index=outer_train.index.copy(),
    )

    return CrossFittedTargetStats(
        train=train_features,
        validation=validation_features,
        feature_names=feature_names,
        inner_folds=tuple(inner_folds),
    )


def _seeded_group_kfold_indices(
    dummy: np.ndarray,
    *,
    groups: np.ndarray,
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Use shuffled GroupKFold, with an equivalent pre-1.6 fallback."""

    try:
        splitter = GroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
    except TypeError:  # pragma: no cover - exercised only on older sklearn.
        unique_groups = pd.Index(pd.unique(groups)).sort_values().to_numpy()
        shuffled_groups = np.random.RandomState(random_state).permutation(unique_groups)
        folds = []
        for valid_groups in np.array_split(shuffled_groups, n_splits):
            valid_mask = np.isin(groups, valid_groups)
            folds.append((np.flatnonzero(~valid_mask), np.flatnonzero(valid_mask)))
        return folds
    return list(splitter.split(dummy, groups=groups))


def _make_fold(
    data: pd.DataFrame,
    group_col: str,
    fold_id: int,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
) -> GroupFold:
    train_idx = np.asarray(train_idx, dtype=int)
    valid_idx = np.asarray(valid_idx, dtype=int)
    return GroupFold(
        fold_id=fold_id,
        train_idx=train_idx,
        valid_idx=valid_idx,
        train_groups=_unique_tuple(data.iloc[train_idx][group_col]),
        valid_groups=_unique_tuple(data.iloc[valid_idx][group_col]),
    )


def _group_feature_matrix(
    data: pd.DataFrame,
    *,
    group_col: str,
    feature_cols: Sequence[str],
    entity_col: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    # A canonical order makes the seeded robustness split independent of the
    # incidental row order.  Sorting here is only identity canonicalisation;
    # it is never used to define a past/future relation.
    group_values = pd.Index(pd.unique(data[group_col]), name=group_col).sort_values()

    numeric = data.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
    working = pd.concat(
        [data[[group_col, *([entity_col] if entity_col else [])]], numeric],
        axis=1,
    )

    if entity_col is None:
        feature_frame = (
            working.groupby(group_col, sort=False, observed=True)[list(feature_cols)]
            .mean()
            .reindex(group_values)
        )
    else:
        matrices = []
        for feature in feature_cols:
            pivot = working.pivot_table(
                index=group_col,
                columns=entity_col,
                values=feature,
                aggfunc="mean",
                sort=True,
            ).reindex(group_values)
            pivot.columns = pd.MultiIndex.from_product([[feature], pivot.columns])
            matrices.append(pivot)
        feature_frame = pd.concat(matrices, axis=1)

    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    # Mean imputation matches the project's earlier feature-balanced protocol.
    feature_frame = feature_frame.fillna(feature_frame.mean()).fillna(0.0)
    matrix = feature_frame.to_numpy(dtype=float)

    minimum = matrix.min(axis=0)
    span = matrix.max(axis=0) - minimum
    scaled = np.zeros_like(matrix, dtype=float)
    varying = span > 0
    scaled[:, varying] = (
        2.0 * (matrix[:, varying] - minimum[varying]) / span[varying] - 1.0
    )
    return group_values.to_numpy(), scaled


def _balanced_assignments(
    matrix: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
) -> list[list[int]]:
    """Reproduce the project's feature-space spread, without dropping groups.

    Each fold starts from a random seed.  A candidate's sampling weight is its
    cumulative squared distance from points already placed in that fold.  The
    old script stopped after ``floor(n_groups / n_splits)`` rounds; this loop
    also assigns the remainder, so validation coverage stays exact.
    """

    n_groups = matrix.shape[0]
    rng = np.random.RandomState(random_state)
    seed_positions = np.sort(
        rng.choice(n_groups, size=n_splits, replace=False)
    )[::-1]

    indexed_matrix = np.column_stack([matrix, np.arange(n_groups)])
    assignments = [[int(position)] for position in seed_positions]
    last_selected = [indexed_matrix[position].copy() for position in seed_positions]

    # Descending seed positions are important: deleting in this order preserves
    # the meaning of every later positional index, matching the legacy code.
    remaining = indexed_matrix.copy()
    for position in seed_positions:
        remaining = np.delete(remaining, int(position), axis=0)

    cumulative_distance = [
        np.zeros(len(remaining), dtype=float) for _ in range(n_splits)
    ]
    while len(remaining):
        thresholds = rng.uniform(0.0, 1.0, size=n_splits)
        for fold_id in range(n_splits):
            if not len(remaining):
                break

            squared_distance = np.square(
                remaining[:, :-1] - last_selected[fold_id][:-1]
            ).sum(axis=1)
            cumulative_distance[fold_id] += squared_distance
            weights = cumulative_distance[fold_id]
            total_weight = float(weights.sum())
            if total_weight > 0.0 and np.isfinite(weights).all():
                cumulative_probability = np.cumsum(weights / total_weight)
                chosen_offset = int(
                    np.searchsorted(
                        cumulative_probability,
                        thresholds[fold_id],
                        side="right",
                    )
                )
                chosen_offset = min(chosen_offset, len(remaining) - 1)
            else:
                chosen_offset = int(rng.randint(len(remaining)))

            chosen_row = remaining[chosen_offset].copy()
            assignments[fold_id].append(int(chosen_row[-1]))
            last_selected[fold_id] = chosen_row
            remaining = np.delete(remaining, chosen_offset, axis=0)
            for other_fold in range(n_splits):
                cumulative_distance[other_fold] = np.delete(
                    cumulative_distance[other_fold], chosen_offset
                )

    return assignments


@dataclass(frozen=True)
class _FittedStockStatistics:
    table: pd.DataFrame
    fallback: dict[str, float]


def _fit_stock_statistics(
    data: pd.DataFrame,
    *,
    stock_col: str,
    target_col: str,
    prefix: str,
) -> _FittedStockStatistics:
    values = data[target_col].to_numpy(dtype=float, copy=False)
    working = pd.DataFrame(
        {stock_col: data[stock_col].to_numpy(copy=False), "_target": values}
    )
    grouped = working.groupby(stock_col, sort=False, observed=True)["_target"]
    table = grouped.agg(["mean", "median", "count"])
    table["std"] = grouped.std(ddof=0)
    table = table.rename(columns={name: f"{prefix}_{name}" for name in table.columns})

    fallback = {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": float(np.std(values, ddof=0)),
        f"{prefix}_count": 0.0,
    }
    return _FittedStockStatistics(table=table, fallback=fallback)


def _apply_stock_statistics(
    data: pd.DataFrame,
    fitted: _FittedStockStatistics,
    *,
    stock_col: str,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    result = pd.DataFrame(index=data.index.copy())
    for name in feature_names:
        result[name] = data[stock_col].map(fitted.table[name]).astype(float)
        result[name] = result[name].fillna(fitted.fallback[name])
    return result


def _finite_target_values(series: pd.Series, name: str) -> np.ndarray:
    try:
        values = series.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name!r} must be numeric") from exc
    if values.size == 0:
        raise ValueError("outer_train must contain at least one row")
    if not np.isfinite(values).all():
        raise ValueError(f"{name!r} must contain only finite values")
    return values


def _checked_indices(indices: np.ndarray, n_rows: int, name: str) -> np.ndarray:
    array = np.asarray(indices)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise AssertionError(f"{name} must be a one-dimensional integer array")
    array = array.astype(int, copy=False)
    if array.size != np.unique(array).size:
        raise AssertionError(f"{name} contains duplicate row positions")
    if array.size and (array.min() < 0 or array.max() >= n_rows):
        raise AssertionError(f"{name} contains an out-of-bounds row position")
    return array


def _validate_n_splits(n_splits: int, n_groups: int) -> None:
    if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)):
        raise TypeError("n_splits must be an integer")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > n_groups:
        raise ValueError(
            f"n_splits={n_splits} exceeds the number of distinct groups ({n_groups})"
        )


def _validate_group_values(series: pd.Series, name: str) -> None:
    _validate_non_null(series, name)
    if series.empty:
        raise ValueError("data must contain at least one row")


def _validate_non_null(series: pd.Series, name: str) -> None:
    if series.isna().any():
        raise ValueError(f"{name!r} must not contain missing values")


def _require_columns(data: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _unique_tuple(values: pd.Series) -> tuple[object, ...]:
    return tuple(pd.unique(values).tolist())


def _short_values(values: set[object], limit: int = 5) -> str:
    ordered = sorted((repr(value) for value in values))
    suffix = " ..." if len(ordered) > limit else ""
    return ", ".join(ordered[:limit]) + suffix


__all__ = [
    "CrossFittedTargetStats",
    "FoldStrategy",
    "GroupFold",
    "assert_valid_group_folds",
    "build_group_folds",
    "cross_fit_stock_target_stats",
]
