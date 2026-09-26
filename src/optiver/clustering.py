"""Leakage-safe stock clustering and same-time peer features.

The original project fitted stock clusters once, using the target values from the
entire data set.  That lets validation targets influence a feature used to score
the validation fold.  The classes in this module deliberately separate ``fit``
from ``transform``: a cluster mapping is learned from an outer training fold and
``transform`` only reads ``stock_id`` from any later frame.

Two clustering strategies are provided:

``TargetCorrelationStockClusterer``
    Reproduces the useful idea from the original project, but builds the target
    correlation matrix from the outer training fold only.

``FeatureStockClusterer``
    A target-free alternative.  It summarizes observable book/trade features by
    stock, standardizes those summaries, and clusters them.

``add_same_time_cluster_aggregates`` computes peer features from observable
columns.  With its default ``leave_one_stock_out=True``, every stock is excluded
from the aggregate for its own cluster.  Duplicate rows for a stock/time pair are
first collapsed, so this is genuinely leave-*one-stock*-out rather than merely
leave-one-row-out.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


ClusterEncoding = Literal["category", "integer", "onehot"]


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _validate_cluster_count(n_clusters: int, n_stocks: int) -> None:
    if not isinstance(n_clusters, int) or isinstance(n_clusters, bool):
        raise TypeError("n_clusters must be an integer")
    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1")
    if n_stocks < n_clusters:
        raise ValueError(
            f"Cannot fit {n_clusters} clusters with only {n_stocks} stocks"
        )


def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    """Return deterministic unique values without requiring comparable types."""

    unique = list(pd.unique(pd.Series(list(values), dtype="object").dropna()))
    try:
        return sorted(unique)
    except TypeError:
        return unique


def _cluster_categories(cluster_labels: Sequence[Any], unknown_label: Any) -> list[Any]:
    categories = list(dict.fromkeys(cluster_labels))
    if unknown_label not in categories:
        categories.append(unknown_label)
    return categories


def _cluster_token(value: Any) -> str:
    """Create a readable, stable token for a generated feature name."""

    token = str(value)
    for old, new in ((" ", "_"), ("/", "_"), ("\\", "_"), (".", "p")):
        token = token.replace(old, new)
    return token


def add_cluster_labels(
    frame: pd.DataFrame,
    stock_to_cluster: Mapping[Any, Any] | pd.Series,
    *,
    stock_col: str = "stock_id",
    cluster_col: str = "stock_cluster",
    encoding: ClusterEncoding = "category",
    cluster_labels: Sequence[Any] | None = None,
    unknown_label: Any = -1,
) -> pd.DataFrame:
    """Return ``frame`` with a fitted stock-cluster mapping applied.

    Parameters
    ----------
    frame:
        Any training, validation, or test frame.  Only ``stock_col`` is read.
    stock_to_cluster:
        Mapping learned before this function is called.  A Series is interpreted
        as ``index=stock_id, value=cluster``.
    encoding:
        ``"category"`` produces a pandas categorical column suitable for
        LightGBM; ``"integer"`` produces a regular label column; ``"onehot"``
        produces stable int8 dummy columns named ``{cluster_col}__{label}``.
    cluster_labels:
        Complete ordered label set.  Passing it ensures identical one-hot columns
        in train and validation even when one fold does not contain every label.
    unknown_label:
        Label assigned to stocks absent from the outer-training mapping.
    """

    _require_columns(frame, [stock_col])
    if encoding not in {"category", "integer", "onehot"}:
        raise ValueError("encoding must be 'category', 'integer', or 'onehot'")

    mapping = (
        stock_to_cluster.to_dict()
        if isinstance(stock_to_cluster, pd.Series)
        else dict(stock_to_cluster)
    )
    inferred_labels = _ordered_unique(mapping.values())
    labels = list(cluster_labels) if cluster_labels is not None else inferred_labels
    categories = _cluster_categories(labels, unknown_label)

    mapped = frame[stock_col].map(mapping).fillna(unknown_label)
    result = frame.copy()

    if encoding == "integer":
        # Preserve arbitrary user labels, while using a compact integer dtype for
        # the common integer-label case.
        if all(isinstance(value, (int, np.integer)) for value in categories):
            result[cluster_col] = mapped.astype(np.int64)
        else:
            result[cluster_col] = mapped
        return result

    categorical = pd.Series(
        pd.Categorical(mapped, categories=categories),
        index=frame.index,
        name=cluster_col,
    )
    if encoding == "category":
        result[cluster_col] = categorical
        return result

    dummies = pd.get_dummies(
        categorical,
        prefix=cluster_col,
        prefix_sep="__",
        dtype=np.int8,
    )
    # pd.get_dummies on a categorical normally emits every category.  Explicit
    # reindexing documents and guarantees the train/validation schema contract.
    expected = [f"{cluster_col}__{label}" for label in categories]
    dummies = dummies.reindex(columns=expected, fill_value=0)
    if cluster_col in result.columns:
        result = result.drop(columns=cluster_col)
    collisions = [column for column in expected if column in result.columns]
    if collisions:
        raise ValueError(f"One-hot columns already exist: {collisions}")
    return pd.concat([result, dummies], axis=1)


class _StockClusterMappingMixin(TransformerMixin):
    """Shared application API for fitted stock-cluster mappings."""

    stock_col: str
    cluster_col: str
    n_clusters: int
    unknown_label: Any

    @property
    def cluster_labels_(self) -> list[int]:
        check_is_fitted(self, "stock_to_cluster_")
        return list(range(self.n_clusters))

    def transform(
        self,
        frame: pd.DataFrame,
        *,
        encoding: ClusterEncoding = "category",
    ) -> pd.DataFrame:
        """Apply the fitted mapping without reading targets or model features."""

        check_is_fitted(self, "stock_to_cluster_")
        return add_cluster_labels(
            frame,
            self.stock_to_cluster_,
            stock_col=self.stock_col,
            cluster_col=self.cluster_col,
            encoding=encoding,
            cluster_labels=self.cluster_labels_,
            unknown_label=self.unknown_label,
        )

    def fit_transform_split(
        self,
        outer_train: pd.DataFrame,
        validation: pd.DataFrame,
        *,
        encoding: ClusterEncoding = "category",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fit on ``outer_train`` and safely apply the mapping to both frames."""

        self.fit(outer_train)
        return (
            self.transform(outer_train, encoding=encoding),
            self.transform(validation, encoding=encoding),
        )


class TargetCorrelationStockClusterer(_StockClusterMappingMixin, BaseEstimator):
    """Cluster stocks by target correlation using one outer training fold only.

    ``fit`` reads the target.  ``transform`` does not: it simply applies the
    learned ``stock_id -> cluster`` dictionary.  Therefore a validation frame may
    omit the target column entirely, and changing validation targets cannot affect
    its cluster labels.
    """

    def __init__(
        self,
        n_clusters: int = 7,
        *,
        stock_col: str = "stock_id",
        time_col: str = "time_id",
        target_col: str = "target",
        cluster_col: str = "stock_cluster",
        min_periods: int = 2,
        correlation_fill_value: float = 0.0,
        unknown_label: Any = -1,
        random_state: int = 42,
        n_init: int | str = 10,
    ) -> None:
        self.n_clusters = n_clusters
        self.stock_col = stock_col
        self.time_col = time_col
        self.target_col = target_col
        self.cluster_col = cluster_col
        self.min_periods = min_periods
        self.correlation_fill_value = correlation_fill_value
        self.unknown_label = unknown_label
        self.random_state = random_state
        self.n_init = n_init

    def fit(
        self,
        outer_train: pd.DataFrame,
        y: Any = None,
    ) -> "TargetCorrelationStockClusterer":
        """Fit exclusively from ``outer_train`` target observations."""

        del y  # Targets must be keyed by stock/time; a detached y is ambiguous.
        _require_columns(
            outer_train, [self.stock_col, self.time_col, self.target_col]
        )
        if self.min_periods < 1:
            raise ValueError("min_periods must be at least 1")
        if outer_train[self.stock_col].isna().any():
            raise ValueError(f"{self.stock_col} must not contain missing values")

        stocks = _ordered_unique(outer_train[self.stock_col])
        _validate_cluster_count(self.n_clusters, len(stocks))

        pivot = outer_train.pivot_table(
            index=self.time_col,
            columns=self.stock_col,
            values=self.target_col,
            aggfunc="mean",
            sort=False,
        ).reindex(columns=stocks)
        if pivot.empty:
            raise ValueError("outer_train contains no usable target observations")

        correlation = pivot.corr(min_periods=self.min_periods).reindex(
            index=stocks, columns=stocks
        )
        correlation = correlation.replace([np.inf, -np.inf], np.nan).fillna(
            self.correlation_fill_value
        )
        matrix = correlation.to_numpy(dtype=float, copy=True)
        # A constant or sparse stock has an undefined self-correlation.  Giving it
        # a valid diagonal still distinguishes it from an all-zero vector.
        np.fill_diagonal(matrix, 1.0)

        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        labels = self.kmeans_.fit_predict(matrix)
        self.stock_ids_ = pd.Index(stocks, name=self.stock_col)
        self.stock_to_cluster_ = dict(zip(stocks, labels.astype(int), strict=True))
        self.target_pivot_ = pivot
        self.correlation_matrix_ = pd.DataFrame(
            matrix, index=self.stock_ids_, columns=self.stock_ids_
        )
        self.n_fit_rows_ = len(outer_train)
        self.n_fit_times_ = pivot.shape[0]
        return self


class FeatureStockClusterer(_StockClusterMappingMixin, BaseEstimator):
    """Target-free stock clustering from observable book/trade features.

    Each feature is summarized per stock with the requested ``aggregations``.
    The resulting stock-level matrix is median-imputed, standardized, and passed
    to KMeans.  By default a column named ``target`` is rejected explicitly.
    """

    _ALLOWED_AGGREGATIONS = {"mean", "median", "std", "min", "max"}

    def __init__(
        self,
        feature_cols: Sequence[str],
        n_clusters: int = 7,
        *,
        stock_col: str = "stock_id",
        cluster_col: str = "stock_cluster",
        aggregations: Sequence[str] = ("mean", "std"),
        target_col: str = "target",
        unknown_label: Any = -1,
        random_state: int = 42,
        n_init: int | str = 10,
    ) -> None:
        self.feature_cols = feature_cols
        self.n_clusters = n_clusters
        self.stock_col = stock_col
        self.cluster_col = cluster_col
        self.aggregations = aggregations
        self.target_col = target_col
        self.unknown_label = unknown_label
        self.random_state = random_state
        self.n_init = n_init

    def fit(
        self,
        outer_train: pd.DataFrame,
        y: Any = None,
    ) -> "FeatureStockClusterer":
        """Fit from outer-training observable features; no target is accessed."""

        del y
        feature_cols = list(self.feature_cols)
        aggregations = list(self.aggregations)
        if not feature_cols:
            raise ValueError("feature_cols must contain at least one column")
        if len(feature_cols) != len(set(feature_cols)):
            raise ValueError("feature_cols must not contain duplicates")
        if not aggregations:
            raise ValueError("aggregations must contain at least one statistic")
        unsupported = set(aggregations) - self._ALLOWED_AGGREGATIONS
        if unsupported:
            raise ValueError(f"Unsupported aggregations: {sorted(unsupported)}")
        if self.target_col in feature_cols:
            raise ValueError(
                f"Refusing to use target column {self.target_col!r} in the "
                "target-free clusterer"
            )
        _require_columns(outer_train, [self.stock_col, *feature_cols])
        if outer_train[self.stock_col].isna().any():
            raise ValueError(f"{self.stock_col} must not contain missing values")

        non_numeric = [
            column
            for column in feature_cols
            if not pd.api.types.is_numeric_dtype(outer_train[column])
        ]
        if non_numeric:
            raise TypeError(f"Clustering features must be numeric: {non_numeric}")

        stocks = _ordered_unique(outer_train[self.stock_col])
        _validate_cluster_count(self.n_clusters, len(stocks))
        summary = outer_train.groupby(self.stock_col, sort=False)[feature_cols].agg(
            aggregations
        )
        summary.columns = [
            f"{feature}__{statistic}" for feature, statistic in summary.columns
        ]
        summary = summary.reindex(stocks).replace([np.inf, -np.inf], np.nan)
        fill_values = summary.median(axis=0).fillna(0.0)
        imputed = summary.fillna(fill_values).astype(float)

        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(imputed)
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        labels = self.kmeans_.fit_predict(scaled)

        self.stock_ids_ = pd.Index(stocks, name=self.stock_col)
        self.stock_to_cluster_ = dict(zip(stocks, labels.astype(int), strict=True))
        self.stock_feature_names_ = list(summary.columns)
        self.stock_feature_fill_values_ = fill_values
        self.stock_feature_summary_ = imputed
        self.n_fit_rows_ = len(outer_train)
        return self


def add_same_time_cluster_aggregates(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    stock_col: str = "stock_id",
    time_col: str = "time_id",
    cluster_col: str = "stock_cluster",
    stats: Sequence[str] = ("mean",),
    clusters: Sequence[Any] | None = None,
    leave_one_stock_out: bool = True,
    minimum_peer_count: int = 1,
    target_col: str = "target",
) -> pd.DataFrame:
    """Add wide, same-time cluster aggregates for observable features.

    For every requested cluster, each row receives columns named
    ``{feature}_cluster_{label}_{stat}``.  A row outside that cluster sees the
    ordinary cluster aggregate.  A row inside it sees an aggregate excluding its
    own stock when ``leave_one_stock_out=True``.

    Aggregation is stock-balanced: duplicate rows for one stock/time are averaged
    first and count as one stock.  Supported statistics are ``mean``, ``sum``, and
    ``count``.  Results with fewer than ``minimum_peer_count`` contributing stocks
    are set to NaN.  The target column is rejected by default to prevent accidental
    construction of contemporaneous target leakage.
    """

    feature_cols = list(feature_cols)
    stats = list(stats)
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one column")
    if len(feature_cols) != len(set(feature_cols)):
        raise ValueError("feature_cols must not contain duplicates")
    unsupported = set(stats) - {"mean", "sum", "count"}
    if not stats or unsupported:
        raise ValueError(
            "stats must contain one or more of 'mean', 'sum', and 'count'; "
            f"unsupported={sorted(unsupported)}"
        )
    if minimum_peer_count < 1:
        raise ValueError("minimum_peer_count must be at least 1")
    if target_col in feature_cols:
        raise ValueError(
            f"Refusing to aggregate target column {target_col!r}; pass only "
            "observable book/trade features"
        )
    _require_columns(
        frame, [stock_col, time_col, cluster_col, *feature_cols]
    )
    non_numeric = [
        column
        for column in feature_cols
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise TypeError(f"Aggregate features must be numeric: {non_numeric}")

    if clusters is None:
        cluster_series = frame[cluster_col]
        if isinstance(cluster_series.dtype, pd.CategoricalDtype):
            cluster_values = list(cluster_series.cat.categories)
        else:
            cluster_values = _ordered_unique(cluster_series)
        # The conventional -1 category denotes an unseen stock.  It is useful as
        # a model label but should not become a synthetic peer group by default.
        cluster_values = [value for value in cluster_values if value != -1]
    else:
        cluster_values = list(dict.fromkeys(clusters))
    if not cluster_values:
        raise ValueError("No non-missing cluster labels are available")

    # One record per stock/time/cluster makes the peer statistic stock-balanced.
    stock_level = (
        frame.groupby(
            [time_col, stock_col, cluster_col],
            observed=True,
            dropna=False,
            sort=False,
        )[feature_cols]
        .mean()
        .reset_index()
    )
    grouped = stock_level.groupby(
        [time_col, cluster_col], observed=True, dropna=False, sort=False
    )[feature_cols].agg(["sum", "count"])

    # Look up each row's stock-level contribution.  This remains correct when the
    # input contains multiple rows for the same stock/time pair.
    stock_keys = pd.MultiIndex.from_frame(
        stock_level[[time_col, stock_col, cluster_col]]
    )
    row_keys = pd.MultiIndex.from_frame(frame[[time_col, stock_col, cluster_col]])
    own_values: dict[str, np.ndarray] = {}
    for feature in feature_cols:
        lookup = pd.Series(stock_level[feature].to_numpy(), index=stock_keys)
        own_values[feature] = lookup.reindex(row_keys).to_numpy(dtype=float)

    result = frame.copy()
    generated_names: set[str] = set()
    for cluster in cluster_values:
        try:
            cluster_group = grouped.xs(cluster, level=cluster_col, drop_level=True)
        except KeyError:
            cluster_group = pd.DataFrame(index=pd.Index([], name=time_col))

        row_is_member = frame[cluster_col].eq(cluster).fillna(False).to_numpy()
        for feature in feature_cols:
            if cluster_group.empty:
                base_sum = np.full(len(frame), np.nan, dtype=float)
                base_count = np.zeros(len(frame), dtype=float)
            else:
                sum_by_time = cluster_group[(feature, "sum")]
                count_by_time = cluster_group[(feature, "count")]
                base_sum = frame[time_col].map(sum_by_time).to_numpy(dtype=float)
                base_count = (
                    frame[time_col]
                    .map(count_by_time)
                    .fillna(0)
                    .to_numpy(dtype=float)
                )

            aggregate_sum = base_sum.copy()
            aggregate_count = base_count.copy()
            if leave_one_stock_out:
                own = own_values[feature]
                subtract = row_is_member & ~np.isnan(own)
                aggregate_sum[subtract] -= own[subtract]
                aggregate_count[subtract] -= 1.0

            enough_peers = aggregate_count >= minimum_peer_count
            for stat in stats:
                name = f"{feature}_cluster_{_cluster_token(cluster)}_{stat}"
                if name in result.columns or name in generated_names:
                    raise ValueError(f"Generated column already exists: {name}")
                generated_names.add(name)

                if stat == "mean":
                    values = np.divide(
                        aggregate_sum,
                        aggregate_count,
                        out=np.full(len(frame), np.nan, dtype=float),
                        where=enough_peers & (aggregate_count > 0),
                    )
                elif stat == "sum":
                    values = aggregate_sum.copy()
                    values[~enough_peers] = np.nan
                else:  # count
                    values = aggregate_count.copy()
                    values[~enough_peers] = np.nan
                result[name] = values

    return result


__all__ = [
    "FeatureStockClusterer",
    "TargetCorrelationStockClusterer",
    "add_cluster_labels",
    "add_same_time_cluster_aggregates",
]
