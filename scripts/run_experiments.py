#!/usr/bin/env python3
"""Run leakage-safe Optiver baselines and LightGBM experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
import sklearn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optiver.baselines import evaluate_baselines  # noqa: E402
from optiver.clustering import (  # noqa: E402
    FeatureStockClusterer,
    TargetCorrelationStockClusterer,
    add_same_time_cluster_aggregates,
)
from optiver.data import load_feature_table, target_derived_columns  # noqa: E402
from optiver.feature_sets import (  # noqa: E402
    FEATURE_GROUPS,
    FEATURE_SET_GROUPS,
    get_feature_set,
)
from optiver.training import (  # noqa: E402
    FeatureProviderScope,
    LightGBMConfig,
    run_lightgbm_cv,
)
from optiver.validation import (  # noqa: E402
    GroupFold,
    build_group_folds,
    cross_fit_stock_target_stats,
)


CLUSTER_SOURCE_COLUMNS = [
    "rv_pred",
    "rv1_last400",
    "rv1_last300",
    "rv1_last200",
    "total_vol_sum",
    "size_sum",
    "trade_count",
    "price_spread_sum",
    "bid_spread_sum",
    "ask_spread_sum",
]
FEATURE_CLUSTER_COLUMNS = [
    "rv_pred",
    "rv1_last400",
    "trade_rv",
    "price_spread_sum",
    "total_vol_sum",
    "size_sum",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["baselines", "lgbm"],
        help="Experiment family to run",
    )
    parser.add_argument("--features-path", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--feature-set",
        choices=["rv_only", *FEATURE_SET_GROUPS],
        default="full_clean",
    )
    parser.add_argument(
        "--exclude-feature-group",
        action="append",
        choices=sorted(FEATURE_GROUPS),
        default=[],
        help=(
            "Repeatable leave-one-group-out control applied after --feature-set "
            "is resolved"
        ),
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--max-rounds", type=int, default=2_000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument(
        "--fixed-rounds",
        type=int,
        help="Skip inner early stopping and use this pre-registered round count",
    )
    parser.add_argument("--num-leaves", type=int, default=64)
    parser.add_argument("--min-data-in-leaf", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument(
        "--lambda-l1",
        type=float,
        default=0.0,
        help="Regularization after RMSPE weights are normalized to mean one",
    )
    parser.add_argument(
        "--lambda-l2",
        type=float,
        default=0.0,
        help="Regularization after RMSPE weights are normalized to mean one",
    )
    parser.add_argument("--weight-clip-quantile", type=float)
    parser.add_argument("--include-stock-id", action="store_true")
    parser.add_argument("--include-stock-target-stats", action="store_true")
    parser.add_argument(
        "--cluster-mode",
        choices=["none", "feature", "target", "oracle_target"],
        default="none",
    )
    parser.add_argument(
        "--allow-oracle",
        action="store_true",
        help="Required for the deliberately contaminated oracle_target diagnostic",
    )
    parser.add_argument("--n-clusters", type=int, default=7)
    parser.add_argument(
        "--split-strategy",
        choices=["group_kfold", "feature_balanced"],
        default="group_kfold",
    )
    parser.add_argument(
        "--shuffle-target-groups-seed",
        type=int,
        help=(
            "Negative control: map time_id target vectors within equal stock-"
            "membership buckets, preserving stock alignment; singleton buckets "
            "are retained and reported"
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _source_file_hashes() -> dict[str, str]:
    """Hash the executable experiment source captured by a run."""

    sources = [
        PROJECT_ROOT / "pyproject.toml",
        Path(__file__).resolve(),
        *sorted((PROJECT_ROOT / "src" / "optiver").glob("*.py")),
    ]
    return {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path)
        for path in sources
        if path.is_file()
    }


def _exclude_feature_groups(
    features: list[str], excluded_groups: list[str]
) -> list[str]:
    excluded = {
        feature
        for group in excluded_groups
        for feature in FEATURE_GROUPS[group]
    }
    selected = [feature for feature in features if feature not in excluded]
    if not selected:
        raise ValueError("Feature-group exclusions removed every selected feature")
    return selected


def _groupwise_shuffle_targets(
    frame: pd.DataFrame,
    *,
    seed: int,
    time_col: str = "time_id",
    stock_col: str = "stock_id",
    target_col: str = "target",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle complete time-group target vectors with stock alignment intact.

    Groups are partitioned by their exact stock-membership signature.  Within
    each signature bucket, a seeded cyclic permutation maps every destination
    ``time_id`` to a different source group.  A singleton signature cannot be
    moved without fabricating or dropping a stock, so it maps to itself and is
    reported as unchanged in the returned audit table.
    """

    required = [time_col, stock_col, target_col]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Cannot shuffle targets; missing columns: {missing}")
    if frame.duplicated([time_col, stock_col]).any():
        raise ValueError(
            f"Group-wise shuffle requires unique ({time_col}, {stock_col}) rows"
        )

    groups_by_signature: dict[tuple[object, ...], list[object]] = {}
    for group, rows in frame.groupby(time_col, sort=False, observed=True):
        signature = tuple(sorted(rows[stock_col].tolist(), key=repr))
        groups_by_signature.setdefault(signature, []).append(group)

    rng = np.random.default_rng(seed)
    mapping_records: list[dict[str, Any]] = []
    for signature_id, (_, groups) in enumerate(groups_by_signature.items()):
        ordered = [groups[index] for index in rng.permutation(len(groups))]
        sources = ordered[1:] + ordered[:1] if len(ordered) > 1 else ordered
        source_by_destination = dict(zip(ordered, sources, strict=True))
        for destination in groups:
            source = source_by_destination[destination]
            mapping_records.append(
                {
                    time_col: destination,
                    "source_time_id": source,
                    "signature_id": signature_id,
                    "signature_size": len(signature),
                    "signature_group_count": len(groups),
                    "changed": bool(destination != source),
                }
            )

    mapping = pd.DataFrame.from_records(mapping_records)
    row_keys = frame[[time_col, stock_col]].copy()
    row_keys["_row_position"] = np.arange(len(frame), dtype=np.int64)
    row_keys = row_keys.merge(
        mapping[[time_col, "source_time_id"]],
        on=time_col,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    source_targets = frame[[time_col, stock_col, target_col]].rename(
        columns={time_col: "source_time_id", target_col: "_shuffled_target"}
    )
    aligned = row_keys.merge(
        source_targets,
        on=["source_time_id", stock_col],
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("_row_position")
    if aligned["_shuffled_target"].isna().any():
        raise RuntimeError("Group-wise shuffle lost stock alignment")

    shuffled = frame.copy()
    shuffled[target_col] = aligned["_shuffled_target"].to_numpy()
    return shuffled, mapping.sort_values("signature_id").reset_index(drop=True)


def _fold_pairs(folds: list[GroupFold]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(fold.train_idx, fold.valid_idx) for fold in folds]


def _fold_assignment(frame: pd.DataFrame, folds: list[GroupFold]) -> pd.DataFrame:
    assignment = pd.Series(-1, index=frame.index, dtype=np.int16)
    for fold in folds:
        assignment.iloc[fold.valid_idx] = fold.fold_id
    result = frame[["time_id"]].copy()
    result["fold"] = assignment.to_numpy()
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


def _inner_split_factory(n_splits: int):
    def make_inner(groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
        inner_frame = pd.DataFrame({"time_id": groups})
        fold = build_group_folds(
            inner_frame,
            group_col="time_id",
            n_splits=n_splits,
            strategy="group_kfold",
            random_state=seed,
        )[0]
        return fold.train_idx, fold.valid_idx

    return make_inner


def _add_stock_id_category(
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    known = sorted(pd.unique(train["stock_id"]).tolist())
    categories = [*known, -1]
    train_values = train["stock_id"].where(train["stock_id"].isin(known), -1)
    valid_values = valid["stock_id"].where(valid["stock_id"].isin(known), -1)
    return (
        pd.Series(pd.Categorical(train_values, categories=categories), index=train.index),
        pd.Series(pd.Categorical(valid_values, categories=categories), index=valid.index),
    )


def _feature_provider(
    base_features: list[str],
    *,
    include_stock_id: bool,
    include_stock_target_stats: bool,
    inner_splits: int,
    cluster_mode: str,
    n_clusters: int,
    seed: int,
    oracle_clusterer: TargetCorrelationStockClusterer | None = None,
):
    def provide(
        outer_train: pd.DataFrame,
        outer_valid: pd.DataFrame,
        fold: int,
        scope: FeatureProviderScope,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        # ``scope`` is intentionally part of the provider contract: during
        # early stopping outer_train is only the raw inner-training partition;
        # during final fitting it is the complete outer-training partition.
        if scope not in {"inner_selection", "outer_refit"}:
            raise ValueError(f"Unknown feature-provider scope: {scope!r}")
        x_train = outer_train[base_features].copy()
        x_valid = outer_valid[base_features].copy()
        categorical: list[str] = []

        if include_stock_id:
            train_stock, valid_stock = _add_stock_id_category(outer_train, outer_valid)
            x_train["stock_id_category"] = train_stock
            x_valid["stock_id_category"] = valid_stock
            categorical.append("stock_id_category")

        if include_stock_target_stats:
            encoded = cross_fit_stock_target_stats(
                outer_train,
                outer_valid.drop(columns=["target"], errors="ignore"),
                inner_splits=inner_splits,
                prefix="stock_target",
            )
            for name in encoded.feature_names:
                x_train[name] = encoded.train[name].to_numpy()
                x_valid[name] = encoded.validation[name].to_numpy()
            x_train["rv_vs_stock_target_mean"] = (
                x_train["rv_pred"] / (x_train["stock_target_mean"] + 1e-12)
            )
            x_valid["rv_vs_stock_target_mean"] = (
                x_valid["rv_pred"] / (x_valid["stock_target_mean"] + 1e-12)
            )

        if cluster_mode != "none":
            if cluster_mode in {"target", "oracle_target"}:
                clusterer = TargetCorrelationStockClusterer(
                    n_clusters=n_clusters,
                    random_state=seed + fold,
                )
            else:
                clusterer = FeatureStockClusterer(
                    FEATURE_CLUSTER_COLUMNS,
                    n_clusters=n_clusters,
                    random_state=seed + fold,
                )
            if cluster_mode == "oracle_target":
                if oracle_clusterer is None:
                    raise RuntimeError("oracle_target requires a pre-fitted global clusterer")
                clustered_train = oracle_clusterer.transform(
                    outer_train, encoding="category"
                )
                clustered_valid = oracle_clusterer.transform(
                    outer_valid.drop(columns=["target"], errors="ignore"),
                    encoding="category",
                )
            else:
                clustered_train, clustered_valid = clusterer.fit_transform_split(
                    outer_train,
                    outer_valid.drop(columns=["target"], errors="ignore"),
                    encoding="category",
                )
            augmented_train = add_same_time_cluster_aggregates(
                clustered_train,
                CLUSTER_SOURCE_COLUMNS,
                clusters=range(n_clusters),
                leave_one_stock_out=True,
            )
            augmented_valid = add_same_time_cluster_aggregates(
                clustered_valid,
                CLUSTER_SOURCE_COLUMNS,
                clusters=range(n_clusters),
                leave_one_stock_out=True,
            )
            generated = [
                column
                for column in augmented_train.columns
                if "_cluster_" in column and column.endswith("_mean")
            ]
            x_train["stock_cluster"] = augmented_train["stock_cluster"].to_numpy()
            x_valid["stock_cluster"] = augmented_valid["stock_cluster"].to_numpy()
            for name in generated:
                x_train[name] = augmented_train[name].to_numpy()
                x_valid[name] = augmented_valid[name].to_numpy()
            categorical.append("stock_cluster")

        prohibited = target_derived_columns(x_train.columns)
        # A newly fold-fitted stock_cluster is safe; cached c1 columns never are.
        prohibited = [
            column
            for column in prohibited
            if not (column == "stock_cluster" and cluster_mode != "none")
        ]
        if prohibited:
            raise RuntimeError(f"Prohibited legacy target-derived columns: {prohibited}")
        return x_train, x_valid, categorical

    return provide


def _run_metadata(args: argparse.Namespace, feature_path: Path) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": vars(args) | {
            "features_path": str(args.features_path),
            "output_root": str(args.output_root),
        },
        "data": {
            "path": str(feature_path),
            "size_bytes": feature_path.stat().st_size,
            "sha256": _sha256(feature_path),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "source_files_sha256": _source_file_hashes(),
    }


def _execute_run(
    args: argparse.Namespace,
    *,
    feature_path: Path,
    output_dir: Path,
) -> None:
    if args.command == "baselines":
        if args.exclude_feature_group:
            raise ValueError("Feature-group exclusions do not apply to baselines")
        needed = ["stock_id", "time_id", "target", "rv_pred"]
        base_features: list[str] = []
    else:
        import pyarrow.parquet as pq

        schema = pq.ParquetFile(feature_path).schema_arrow.names
        base_features = get_feature_set(args.feature_set, schema, strict=True)
        base_features = _exclude_feature_groups(
            base_features, args.exclude_feature_group
        )
        needed = ["stock_id", "time_id", "target", *base_features]
        if args.cluster_mode != "none":
            needed.extend(CLUSTER_SOURCE_COLUMNS)
            if args.cluster_mode == "feature":
                needed.extend(FEATURE_CLUSTER_COLUMNS)
        needed = list(dict.fromkeys(needed))

    frame = load_feature_table(feature_path, columns=needed)
    folds = _make_folds(
        frame,
        n_splits=args.n_splits,
        seed=args.seed,
        strategy=args.split_strategy,
    )
    fold_map = _fold_assignment(frame, folds)
    fold_map_path = output_dir / "fold_assignments.csv"
    fold_map.to_csv(fold_map_path, index=False)

    target_shuffle_metadata: dict[str, Any] = {"enabled": False}
    if args.shuffle_target_groups_seed is not None:
        frame, target_group_mapping = _groupwise_shuffle_targets(
            frame, seed=args.shuffle_target_groups_seed
        )
        mapping_path = output_dir / "target_group_permutation.csv"
        target_group_mapping.to_csv(mapping_path, index=False)
        target_shuffle_metadata = {
            "enabled": True,
            "seed": args.shuffle_target_groups_seed,
            "definition": (
                "time_id vectors are cyclically permuted within exact "
                "stock-membership signatures; target values remain aligned by stock_id"
            ),
            "mapping_sha256": _sha256(mapping_path),
            "groups": int(len(target_group_mapping)),
            "changed_groups": int(target_group_mapping["changed"].sum()),
            "singleton_signature_groups": int(
                (target_group_mapping["signature_group_count"] == 1).sum()
            ),
        }

    metadata = _run_metadata(args, feature_path)
    metadata["rows"] = len(frame)
    metadata["stocks"] = int(frame["stock_id"].nunique())
    metadata["time_ids"] = int(frame["time_id"].nunique())
    metadata["fold_map"] = {
        "path": fold_map_path.name,
        "sha256": _sha256(fold_map_path),
        "rows": int(len(fold_map)),
    }
    metadata["target_shuffle"] = target_shuffle_metadata

    if args.command == "baselines":
        metadata["effective_parameters"] = {
            "experiment": "baselines",
            "validation": {
                "n_splits": args.n_splits,
                "split_strategy": args.split_strategy,
                "seed": args.seed,
                "group": "time_id",
            },
        }
        _json_dump(output_dir / "run_metadata.json", metadata)
        result = evaluate_baselines(
            frame,
            _fold_pairs(folds),
            seed=args.seed,
        )
        result.predictions.to_parquet(output_dir / "oof_predictions.parquet", index=False)
        result.fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
        result.summary.to_csv(output_dir / "summary.csv", index=False)
        print(result.summary.to_string(index=False))
        return

    params = {
        "objective": "regression",
        "metric": "None",
        "boosting_type": "gbdt",
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": args.min_data_in_leaf,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "verbosity": -1,
        "num_threads": args.num_threads,
        "deterministic": True,
        "force_col_wise": True,
    }
    config = LightGBMConfig(
        params=params,
        max_boost_rounds=args.max_rounds,
        early_stopping_rounds=args.early_stopping_rounds,
        fixed_boost_rounds=args.fixed_rounds,
        weight_clip_quantile=args.weight_clip_quantile,
        seed=args.seed,
    )
    metadata["effective_parameters"] = {
        "experiment": "lightgbm",
        "base_features": base_features,
        "excluded_feature_groups": list(args.exclude_feature_group),
        "validation": {
            "outer_n_splits": args.n_splits,
            "inner_n_splits": args.inner_splits,
            "split_strategy": args.split_strategy,
            "group": "time_id",
            "seed": args.seed,
        },
        "feature_provider": {
            "include_stock_id": args.include_stock_id,
            "include_stock_target_stats": args.include_stock_target_stats,
            "cluster_mode": args.cluster_mode,
            "n_clusters": args.n_clusters,
            "fit_scope_protocol": (
                "inner_selection fits raw inner-train only; outer_refit rebuilds "
                "features from complete outer-train; validation targets are hidden"
            ),
        },
        "lightgbm_params": params,
        "training": {
            "max_boost_rounds": config.max_boost_rounds,
            "early_stopping_rounds": config.early_stopping_rounds,
            "fixed_boost_rounds": config.fixed_boost_rounds,
            "weight_clip_quantile": config.weight_clip_quantile,
            "fold_seed_rule": "seed + zero_based_outer_fold",
        },
    }
    _json_dump(output_dir / "run_metadata.json", metadata)
    oracle_clusterer = None
    if args.cluster_mode == "oracle_target":
        oracle_clusterer = TargetCorrelationStockClusterer(
            n_clusters=args.n_clusters,
            random_state=args.seed,
        ).fit(frame)
    result = run_lightgbm_cv(
        frame,
        _fold_pairs(folds),
        feature_provider=_feature_provider(
            base_features,
            include_stock_id=args.include_stock_id,
            include_stock_target_stats=args.include_stock_target_stats,
            inner_splits=args.inner_splits,
            cluster_mode=args.cluster_mode,
            n_clusters=args.n_clusters,
            seed=args.seed,
            oracle_clusterer=oracle_clusterer,
        ),
        inner_split_factory=_inner_split_factory(args.inner_splits),
        config=config,
        model_dir=output_dir / "models",
    )
    oof = frame[["stock_id", "time_id", "target"]].copy()
    oof["fold"] = result.fold_ids
    oof["prediction"] = result.oof_predictions
    oof.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    result.fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    result.importance.to_csv(output_dir / "feature_importance.csv", index=False)
    _json_dump(output_dir / "summary.json", result.summary)
    _json_dump(output_dir / "feature_manifest.json", result.feature_names)
    print(json.dumps(result.summary, indent=2, sort_keys=True))


def main() -> None:
    args = _parse_args()
    if args.cluster_mode == "oracle_target" and not args.allow_oracle:
        raise ValueError("oracle_target is contaminated by design; pass --allow-oracle")
    if (
        args.cluster_mode == "oracle_target"
        and args.shuffle_target_groups_seed is not None
    ):
        raise ValueError("Do not combine oracle_target with the label-shuffle control")
    feature_path = args.features_path.expanduser().resolve()
    output_dir = args.output_root.expanduser().resolve() / args.run_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_dir}")
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
        completed_at = datetime.now(timezone.utc).isoformat()
        completion = {
            "status": "completed",
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "run_metadata_sha256": _sha256(output_dir / "run_metadata.json"),
        }
        _json_dump(output_dir / "COMPLETED.json", completion)
        _json_dump(status_path, completion)


if __name__ == "__main__":
    main()
