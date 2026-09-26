"""Leakage-safe MLP training primitives for the Optiver experiments.

PyTorch is imported lazily so the data and validation helpers remain usable in
the lightweight base environment.  The outer validation fold is used exactly
once, for the final score.  When ``fixed_epochs`` is not supplied, the epoch
count is selected on a group-preserving inner split and the model is then
retrained from scratch on the complete outer-training fold.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import target_derived_columns
from .metrics import grouped_bootstrap_rmspe, rmspe, validate_positive_targets
from .validation import build_group_folds


IndexPair = tuple[np.ndarray, np.ndarray]


# A deliberately compact, target-free subset of the engineered features.  It
# keeps the MLP complementary to the full-feature LightGBM while retaining the
# strongest volatility, order-book, and trade signals from the original MLP.
CURATED_MLP_FEATURES = [
    "rv_pred",
    "bpv_sqrt",
    "rv1_last300",
    "rv_accel",
    "obi_mean",
    "obi_std",
    "obi_last",
    "depth_imb_mean",
    "wap_range",
    "wap1_std",
    "wap2_std",
    "wap_balance_sum",
    "wap_balance_max",
    "price_spread_sum",
    "price_spread2_sum",
    "bid_spread_sum",
    "ask_spread_sum",
    "bid_ask_sp_sum",
    "bid_ask_sp_max",
    "total_vol_sum",
    "vol_imb_sum",
    "book_count",
    "trade_rv",
    "trade_count",
    "trade_count_max",
    "size_sum",
    "size_max",
    "amount_sum",
    "unique_seconds",
    "size_tau",
    "size_tau2",
    "size_tau2_d",
    "trade_vol_accel",
    "tendency",
    "df_max",
    "df_min",
    "price_iqr",
    "size_iqr",
    "f_max",
    "f_min",
    "rv_cs_rank",
    "spread_cs_rank",
    "mkt_rv_pred_mean",
    "mkt_rv_pred_std",
]


@dataclass(frozen=True)
class NumericPreprocessor:
    """Median imputation and z-scoring fitted on one training partition."""

    feature_names: tuple[str, ...]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    clip_value: float | None = 12.0

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        feature_names: Sequence[str],
        *,
        clip_value: float | None = 12.0,
    ) -> "NumericPreprocessor":
        names = _validate_feature_names(frame, feature_names)
        if clip_value is not None and clip_value <= 0:
            raise ValueError("clip_value must be positive or None")
        values = _numeric_matrix(frame, names)
        medians = np.zeros(values.shape[1], dtype=np.float64)
        for column in range(values.shape[1]):
            finite = values[np.isfinite(values[:, column]), column]
            medians[column] = float(np.median(finite)) if finite.size else 0.0
        imputed = _impute(values, medians)
        means = imputed.mean(axis=0, dtype=np.float64)
        scales = imputed.std(axis=0, dtype=np.float64)
        scales[~np.isfinite(scales) | (scales < 1e-12)] = 1.0
        return cls(
            feature_names=names,
            medians=medians,
            means=means,
            scales=scales,
            clip_value=clip_value,
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        names = _validate_feature_names(frame, self.feature_names)
        if names != self.feature_names:
            raise ValueError("Feature order differs from the fitted preprocessor")
        values = _impute(_numeric_matrix(frame, names), self.medians)
        transformed = (values - self.means) / self.scales
        if self.clip_value is not None:
            transformed = np.clip(
                transformed, -self.clip_value, self.clip_value
            )
        if not np.all(np.isfinite(transformed)):
            raise ValueError("Preprocessing produced NaN or infinite values")
        return transformed.astype(np.float32, copy=False)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            feature_names=np.asarray(self.feature_names, dtype=str),
            medians=self.medians,
            means=self.means,
            scales=self.scales,
            clip_value=np.asarray(
                np.nan if self.clip_value is None else self.clip_value
            ),
        )


@dataclass(frozen=True)
class StockVocabulary:
    """Outer-train-only stock mapping with zero reserved for unknown stocks."""

    stocks: tuple[object, ...]

    @classmethod
    def fit(cls, values: Sequence[object] | pd.Series) -> "StockVocabulary":
        series = pd.Series(values, copy=False)
        if series.isna().any():
            raise ValueError("stock_id contains missing values")
        # repr is a stable tie-breaker for mixed but hashable scalar types.
        unique = sorted(pd.unique(series).tolist(), key=lambda value: repr(value))
        return cls(tuple(unique))

    @property
    def size_with_unknown(self) -> int:
        return len(self.stocks) + 1

    def encode(self, values: Sequence[object] | pd.Series) -> np.ndarray:
        lookup = {value: index + 1 for index, value in enumerate(self.stocks)}
        series = pd.Series(values, copy=False)
        encoded = series.map(lookup).fillna(0).to_numpy(dtype=np.int64, copy=True)
        return encoded

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        values = [_python_scalar(value) for value in self.stocks]
        destination.write_text(
            json.dumps({"unknown_index": 0, "stocks": values}, indent=2),
            encoding="utf-8",
        )


@dataclass(frozen=True)
class PreparedFeatures:
    train_numeric: np.ndarray
    valid_numeric: np.ndarray
    train_stock_ids: np.ndarray
    valid_stock_ids: np.ndarray
    preprocessor: NumericPreprocessor
    stock_vocabulary: StockVocabulary


def prepare_fold_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    stock_col: str = "stock_id",
    clip_value: float | None = 12.0,
) -> PreparedFeatures:
    """Fit every learned feature transform on ``train`` and apply to ``valid``.

    The function has no target argument and never reads a target column.  This
    structural separation makes it suitable for both the inner split and the
    final outer-fold preprocessing.
    """

    if stock_col not in train or stock_col not in valid:
        raise ValueError(f"Both frames must contain {stock_col!r}")
    preprocessor = NumericPreprocessor.fit(
        train, feature_names, clip_value=clip_value
    )
    vocabulary = StockVocabulary.fit(train[stock_col])
    return PreparedFeatures(
        train_numeric=preprocessor.transform(train),
        valid_numeric=preprocessor.transform(valid),
        train_stock_ids=vocabulary.encode(train[stock_col]),
        valid_stock_ids=vocabulary.encode(valid[stock_col]),
        preprocessor=preprocessor,
        stock_vocabulary=vocabulary,
    )


def _default_hidden_dims() -> tuple[int, ...]:
    return (256, 128, 64)


@dataclass(frozen=True)
class MLPConfig:
    hidden_dims: tuple[int, ...] = field(default_factory=_default_hidden_dims)
    embedding_dim: int = 16
    dropout: float = 0.20
    batch_size: int = 4_096
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20
    min_delta: float = 1e-5
    fixed_epochs: int | None = None
    inner_splits: int = 5
    gradient_clip_norm: float = 1.0
    feature_clip_value: float | None = 12.0
    seed: int = 2021
    device: str = "auto"
    num_workers: int = 0


@dataclass
class MLPCVResult:
    oof_predictions: np.ndarray
    fold_ids: np.ndarray
    fold_metrics: pd.DataFrame
    training_history: pd.DataFrame
    summary: dict[str, float | int | str]
    feature_names: list[str]
    resolved_device: str


def resolve_device(requested: str = "auto") -> str:
    """Resolve ``auto`` to CUDA, MPS, or CPU after checking availability."""

    torch, _ = _import_torch()
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    return requested


def run_mlp_cv(
    frame: pd.DataFrame,
    outer_folds: Sequence[IndexPair],
    feature_names: Sequence[str],
    *,
    config: MLPConfig | None = None,
    target_col: str = "target",
    stock_col: str = "stock_id",
    time_col: str = "time_id",
    model_dir: str | Path | None = None,
    allow_partial_oof: bool = False,
) -> MLPCVResult:
    """Run nested group CV and return auditable predictions.

    ``allow_partial_oof`` exists only for fast pilot runs.  Production runs
    keep the default and must validate every input row exactly once.
    """

    if config is None:
        config = MLPConfig()
    _validate_config(config)
    names = list(_validate_feature_names(frame, feature_names))
    prohibited = set(target_derived_columns(names))
    prohibited.update({target_col, stock_col, time_col}.intersection(names))
    if prohibited:
        raise ValueError(
            "MLP features contain labels, identifiers, or cached target-derived "
            f"columns: {sorted(prohibited)}"
        )
    for column in (target_col, stock_col, time_col):
        if column not in frame:
            raise ValueError(f"frame is missing required column {column!r}")
    targets_all = validate_positive_targets(
        frame[target_col].to_numpy(dtype=np.float64)
    )
    groups_all = frame[time_col].to_numpy()
    folds = _validate_outer_folds(
        outer_folds,
        groups_all,
        len(frame),
        require_complete=not allow_partial_oof,
    )
    device = resolve_device(config.device)
    torch, _ = _import_torch()

    oof = np.full(len(frame), np.nan, dtype=np.float64)
    fold_ids = np.full(len(frame), -1, dtype=np.int16)
    fold_records: list[dict[str, float | int | str]] = []
    history_records: list[dict[str, float | int | str]] = []
    output_models = Path(model_dir) if model_dir is not None else None
    if output_models is not None:
        output_models.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, valid_idx) in enumerate(folds):
        outer_train = frame.iloc[train_idx].reset_index(drop=True)
        outer_valid = frame.iloc[valid_idx].reset_index(drop=True)
        y_train = outer_train[target_col].to_numpy(dtype=np.float32)

        if config.fixed_epochs is None:
            inner_fold = build_group_folds(
                outer_train[[time_col]],
                group_col=time_col,
                n_splits=config.inner_splits,
                strategy="group_kfold",
                random_state=config.seed + fold,
            )[0]
            inner_train = outer_train.iloc[inner_fold.train_idx].reset_index(drop=True)
            inner_valid = outer_train.iloc[inner_fold.valid_idx].reset_index(drop=True)
            inner_prepared = prepare_fold_features(
                inner_train,
                inner_valid,
                names,
                stock_col=stock_col,
                clip_value=config.feature_clip_value,
            )
            inner_seed = config.seed + 10_000 + fold
            _, inner_history, selected_epochs, inner_score = _fit_model(
                inner_prepared.train_stock_ids,
                inner_prepared.train_numeric,
                inner_train[target_col].to_numpy(dtype=np.float32),
                n_stock_embeddings=inner_prepared.stock_vocabulary.size_with_unknown,
                config=config,
                device=device,
                seed=inner_seed,
                max_epochs=config.max_epochs,
                valid_data=(
                    inner_prepared.valid_stock_ids,
                    inner_prepared.valid_numeric,
                    inner_valid[target_col].to_numpy(dtype=np.float32),
                ),
            )
            for record in inner_history:
                history_records.append(
                    {"fold": fold, "phase": "inner_selection", **record}
                )
        else:
            selected_epochs = int(config.fixed_epochs)
            inner_score = float("nan")

        prepared = prepare_fold_features(
            outer_train,
            outer_valid,
            names,
            stock_col=stock_col,
            clip_value=config.feature_clip_value,
        )
        outer_seed = config.seed + fold
        model, outer_history, _, _ = _fit_model(
            prepared.train_stock_ids,
            prepared.train_numeric,
            y_train,
            n_stock_embeddings=prepared.stock_vocabulary.size_with_unknown,
            config=config,
            device=device,
            seed=outer_seed,
            max_epochs=selected_epochs,
            valid_data=None,
        )
        for record in outer_history:
            history_records.append({"fold": fold, "phase": "outer_fit", **record})

        predictions = _predict(
            model,
            prepared.valid_stock_ids,
            prepared.valid_numeric,
            batch_size=config.batch_size * 2,
            device=device,
        )
        # The outer-fold labels are first materialized only after prediction;
        # they cannot influence transforms, epoch selection, or fitting.
        y_valid = outer_valid[target_col].to_numpy(dtype=np.float32)
        outer_score = rmspe(y_valid.astype(np.float64), predictions)
        oof[valid_idx] = predictions
        fold_ids[valid_idx] = fold
        fold_records.append(
            {
                "fold": fold,
                "train_rows": len(train_idx),
                "valid_rows": len(valid_idx),
                "train_time_ids": outer_train[time_col].nunique(),
                "valid_time_ids": outer_valid[time_col].nunique(),
                "selected_epochs": selected_epochs,
                "inner_rmspe": inner_score,
                "outer_rmspe": outer_score,
                "device": device,
                "seed": outer_seed,
            }
        )

        if output_models is not None:
            state = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }
            torch.save(
                {
                    "state_dict": state,
                    "feature_names": names,
                    "n_stock_embeddings": prepared.stock_vocabulary.size_with_unknown,
                    "config": asdict(config),
                    "selected_epochs": selected_epochs,
                },
                output_models / f"fold_{fold}.pt",
            )
            prepared.preprocessor.save(
                output_models / f"fold_{fold}_preprocessor.npz"
            )
            prepared.stock_vocabulary.save(
                output_models / f"fold_{fold}_stock_vocabulary.json"
            )

    oof_mask = fold_ids >= 0
    if (not allow_partial_oof and np.any(~oof_mask)) or np.any(
        ~np.isfinite(oof[oof_mask])
    ):
        missing = int(np.count_nonzero(~oof_mask))
        raise RuntimeError(
            "Outer folds did not produce exactly one prediction per row; "
            f"missing={missing}"
        )
    if not np.any(oof_mask):
        raise RuntimeError("No OOF predictions were produced")
    fold_metrics = pd.DataFrame(fold_records)
    history = pd.DataFrame(history_records)
    interval = grouped_bootstrap_rmspe(
        targets_all[oof_mask],
        oof[oof_mask],
        groups_all[oof_mask],
        n_resamples=1_000,
        seed=config.seed,
    )
    summary: dict[str, float | int | str] = {
        "n_rows": len(frame),
        "n_oof_rows": int(oof_mask.sum()),
        "oof_fraction": float(oof_mask.mean()),
        "partial_run": bool(allow_partial_oof),
        "n_features": len(names),
        "n_folds": len(folds),
        "mean_fold_rmspe": float(fold_metrics["outer_rmspe"].mean()),
        "std_fold_rmspe": float(fold_metrics["outer_rmspe"].std(ddof=0)),
        "pooled_oof_rmspe": interval.estimate,
        "group_bootstrap_ci_lower": interval.lower,
        "group_bootstrap_ci_upper": interval.upper,
        "resolved_device": device,
    }
    return MLPCVResult(
        oof_predictions=oof,
        fold_ids=fold_ids,
        fold_metrics=fold_metrics,
        training_history=history,
        summary=summary,
        feature_names=names,
        resolved_device=device,
    )


def _fit_model(
    stock_ids: np.ndarray,
    numeric: np.ndarray,
    targets: np.ndarray,
    *,
    n_stock_embeddings: int,
    config: MLPConfig,
    device: str,
    seed: int,
    max_epochs: int,
    valid_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[Any, list[dict[str, float | int]], int, float]:
    torch, nn = _import_torch()
    _seed_everything(torch, seed)
    initial_target = float(np.median(targets))
    model = _build_model(
        torch,
        nn,
        n_stock_embeddings=n_stock_embeddings,
        n_numeric=numeric.shape[1],
        embedding_dim=config.embedding_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        initial_target=initial_target,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(stock_ids, dtype=torch.long),
        torch.as_tensor(numeric, dtype=torch.float32),
        torch.as_tensor(targets, dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
        pin_memory=device == "cuda",
    )

    history: list[dict[str, float | int]] = []
    best_score = float("inf")
    best_epoch = 1
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        squared_error_sum = 0.0
        sample_count = 0
        for batch_stock, batch_numeric, batch_target in loader:
            batch_stock = batch_stock.to(device)
            batch_numeric = batch_numeric.to(device)
            batch_target = batch_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_stock, batch_numeric)
            relative = (prediction - batch_target) / torch.clamp(
                batch_target, min=1e-8
            )
            loss = torch.mean(torch.square(relative))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            optimizer.step()
            squared_error_sum += float(loss.detach().cpu()) * len(batch_target)
            sample_count += len(batch_target)
        train_rmspe = math.sqrt(squared_error_sum / sample_count)
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_rmspe_objective": train_rmspe,
        }

        if valid_data is not None:
            valid_prediction = _predict(
                model,
                valid_data[0],
                valid_data[1],
                batch_size=config.batch_size * 2,
                device=device,
            )
            valid_score = rmspe(valid_data[2], valid_prediction)
            record["valid_rmspe"] = valid_score
            if valid_score < best_score - config.min_delta:
                best_score = valid_score
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
        history.append(record)
        if valid_data is not None and stale_epochs >= config.patience:
            break

    if valid_data is None:
        best_epoch = max_epochs
        best_score = float("nan")
    return model, history, best_epoch, best_score


def _predict(
    model: Any,
    stock_ids: np.ndarray,
    numeric: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    torch, _ = _import_torch()
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(stock_ids), batch_size):
            stop = start + batch_size
            batch_stock = torch.as_tensor(
                stock_ids[start:stop], dtype=torch.long, device=device
            )
            batch_numeric = torch.as_tensor(
                numeric[start:stop], dtype=torch.float32, device=device
            )
            prediction = model(batch_stock, batch_numeric)
            predictions.append(prediction.detach().cpu().numpy())
    result = np.concatenate(predictions).astype(np.float64, copy=False)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("MLP produced NaN or infinite predictions")
    return np.clip(result, 0.0, None)


def _build_model(
    torch: Any,
    nn: Any,
    *,
    n_stock_embeddings: int,
    n_numeric: int,
    embedding_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
    initial_target: float,
) -> Any:
    class VolatilityMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stock_embedding = nn.Embedding(
                n_stock_embeddings, embedding_dim, padding_idx=0
            )
            layers: list[Any] = []
            width = n_numeric + embedding_dim
            for hidden in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(width, hidden),
                        nn.LayerNorm(hidden),
                        nn.SiLU(),
                        nn.Dropout(dropout),
                    ]
                )
                width = hidden
            self.backbone = nn.Sequential(*layers)
            self.output = nn.Linear(width, 1)
            self.positive = nn.Softplus()
            nn.init.normal_(self.stock_embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                self.stock_embedding.weight[0].zero_()
            nn.init.zeros_(self.output.weight)
            initial = max(float(initial_target), 1e-8)
            inverse_softplus = math.log(math.expm1(initial))
            nn.init.constant_(self.output.bias, inverse_softplus)

        def forward(self, stock: Any, numeric: Any) -> Any:
            embedded = self.stock_embedding(stock)
            combined = torch.cat([numeric, embedded], dim=1)
            hidden = self.backbone(combined)
            return self.positive(self.output(hidden).squeeze(1)) + 1e-8

    return VolatilityMLP()


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _import_torch() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - depends on local extras
        raise RuntimeError(
            "PyTorch is required for MLP training. Install the optional "
            "dependency with `pip install -e '.[mlp]'`."
        ) from exc
    return torch, nn


def _validate_config(config: MLPConfig) -> None:
    if not config.hidden_dims or any(width < 1 for width in config.hidden_dims):
        raise ValueError("hidden_dims must contain positive widths")
    if config.embedding_dim < 1:
        raise ValueError("embedding_dim must be positive")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if config.batch_size < 1 or config.max_epochs < 1:
        raise ValueError("batch_size and max_epochs must be positive")
    if config.patience < 1 or config.inner_splits < 2:
        raise ValueError("patience must be positive and inner_splits at least 2")
    if config.fixed_epochs is not None and config.fixed_epochs < 1:
        raise ValueError("fixed_epochs must be positive")
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if config.gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    if config.num_workers < 0:
        raise ValueError("num_workers cannot be negative")


def _validate_outer_folds(
    outer_folds: Sequence[IndexPair],
    groups: np.ndarray,
    n_rows: int,
    *,
    require_complete: bool = True,
) -> list[IndexPair]:
    if not outer_folds:
        raise ValueError("At least one outer fold is required")
    owners = np.full(n_rows, -1, dtype=np.int16)
    result: list[IndexPair] = []
    all_rows = np.arange(n_rows)
    for fold, (raw_train, raw_valid) in enumerate(outer_folds):
        train = _validated_positions(raw_train, n_rows, "outer_train")
        valid = _validated_positions(raw_valid, n_rows, "outer_valid")
        if np.intersect1d(train, valid).size:
            raise ValueError(f"Outer fold {fold} overlaps train and validation")
        if not np.array_equal(np.sort(np.concatenate([train, valid])), all_rows):
            raise ValueError(f"Outer fold {fold} is not a row-wise complement")
        if set(groups[train]).intersection(groups[valid]):
            raise ValueError(f"Outer fold {fold} leaks time_id groups")
        if np.any(owners[valid] >= 0):
            raise ValueError(f"Outer fold {fold} reuses validation rows")
        owners[valid] = fold
        result.append((train, valid))
    if require_complete and np.any(owners < 0):
        raise ValueError("Outer folds do not validate every row exactly once")
    return result


def _validated_positions(values: np.ndarray, n_rows: int, name: str) -> np.ndarray:
    positions = np.asarray(values, dtype=int)
    if positions.ndim != 1 or positions.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if np.any(positions < 0) or np.any(positions >= n_rows):
        raise ValueError(f"{name} contains out-of-range positions")
    if np.unique(positions).size != positions.size:
        raise ValueError(f"{name} contains duplicate positions")
    return positions


def _validate_feature_names(
    frame: pd.DataFrame, feature_names: Sequence[str]
) -> tuple[str, ...]:
    names = tuple(feature_names)
    if not names:
        raise ValueError("feature_names cannot be empty")
    if len(set(names)) != len(names):
        raise ValueError("feature_names contains duplicates")
    missing = [name for name in names if name not in frame]
    if missing:
        raise ValueError(f"Missing numeric features: {missing}")
    return names


def _numeric_matrix(frame: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    converted = [
        pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
        for name in names
    ]
    values = np.column_stack(converted)
    values[~np.isfinite(values)] = np.nan
    return values


def _impute(values: np.ndarray, medians: np.ndarray) -> np.ndarray:
    result = values.copy()
    missing_row, missing_column = np.where(~np.isfinite(result))
    result[missing_row, missing_column] = medians[missing_column]
    return result


def _python_scalar(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


__all__ = [
    "CURATED_MLP_FEATURES",
    "MLPConfig",
    "MLPCVResult",
    "NumericPreprocessor",
    "PreparedFeatures",
    "StockVocabulary",
    "prepare_fold_features",
    "resolve_device",
    "run_mlp_cv",
]
