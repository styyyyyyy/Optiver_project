"""Named, auditable feature groups used by the ablation study."""

from __future__ import annotations

from collections.abc import Iterable


CURRENT_WINDOW_RV = ["rv_pred", "rv2", "rv3", "rv4"]

MULTISCALE_RV = [
    "rv1_last500",
    "rv2_last500",
    "rv1_last400",
    "rv2_last400",
    "rv1_last300",
    "rv2_last300",
    "rv1_last200",
    "rv2_last200",
    "rv1_last100",
    "rv2_last100",
    "rv_accel",
]

JUMP_FEATURES = ["bpv", "bpv_sqrt", "jump_var", "jump_frac"]

BOOK_MICROSTRUCTURE = [
    "wap1_std",
    "wap2_std",
    "wap_balance_sum",
    "wap_balance_max",
    "price_spread_sum",
    "price_spread_max",
    "price_spread2_sum",
    "bid_spread_sum",
    "bid_spread_max",
    "ask_spread_sum",
    "ask_spread_max",
    "bid_ask_sp_sum",
    "bid_ask_sp_max",
    "total_vol_sum",
    "total_vol_max",
    "vol_imb_sum",
    "vol_imb_max",
    "obi_mean",
    "obi_std",
    "obi_last",
    "depth_imb_mean",
    "wap_range",
    "book_count",
]

TRADE_FEATURES = [
    "trade_rv",
    "trade_count",
    "trade_count_max",
    "size_sum",
    "size_max",
    "size_min",
    "amount_sum",
    "amount_max",
    "unique_seconds",
    "size_tau",
    "size_tau2",
    "trade_rv_last400",
    "trade_count_last400",
    "size_sum_last400",
    "uniq_sec_last400",
    "size_tau_last400",
    "size_tau2_last400",
    "trade_rv_last300",
    "trade_count_last300",
    "size_sum_last300",
    "uniq_sec_last300",
    "size_tau_last300",
    "size_tau2_last300",
    "trade_rv_last200",
    "trade_count_last200",
    "size_sum_last200",
    "uniq_sec_last200",
    "size_tau_last200",
    "size_tau2_last200",
    "size_tau2_d",
    "trade_vol_accel",
    "tendency",
    "f_max",
    "f_min",
    "df_max",
    "df_min",
    "price_iqr",
    "size_iqr",
]

MARKET_FEATURES = [
    "rv_cs_rank",
    "spread_cs_rank",
    "mkt_rv_pred_mean",
    "mkt_rv_pred_std",
    "mkt_rv2_mean",
    "mkt_rv2_std",
    "mkt_rv1_last400_mean",
    "mkt_rv1_last400_std",
    "mkt_rv1_last300_mean",
    "mkt_rv1_last300_std",
    "mkt_rv1_last200_mean",
    "mkt_rv1_last200_std",
]


FEATURE_GROUPS = {
    "current_window_rv": CURRENT_WINDOW_RV,
    "multiscale_rv": MULTISCALE_RV,
    "jump": JUMP_FEATURES,
    "book_microstructure": BOOK_MICROSTRUCTURE,
    "trade": TRADE_FEATURES,
    "market": MARKET_FEATURES,
}


FEATURE_SET_GROUPS = {
    "current_rv": ["current_window_rv"],
    "multiscale": ["current_window_rv", "multiscale_rv"],
    "jump": ["current_window_rv", "multiscale_rv", "jump"],
    "book": [
        "current_window_rv",
        "multiscale_rv",
        "jump",
        "book_microstructure",
    ],
    "trade": [
        "current_window_rv",
        "multiscale_rv",
        "jump",
        "book_microstructure",
        "trade",
    ],
    "full_clean": [
        "current_window_rv",
        "multiscale_rv",
        "jump",
        "book_microstructure",
        "trade",
        "market",
    ],
}


def _deduplicate(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def get_feature_set(
    name: str,
    available_columns: Iterable[str],
    *,
    strict: bool = True,
) -> list[str]:
    """Resolve a named cumulative feature set against an input schema."""

    if name not in FEATURE_SET_GROUPS:
        if name != "rv_only":
            raise KeyError(
                f"Unknown feature set {name!r}; choose from "
                f"{sorted([*FEATURE_SET_GROUPS, 'rv_only'])}"
            )
    requested = (
        ["rv_pred"]
        if name == "rv_only"
        else _deduplicate(
            feature
            for group_name in FEATURE_SET_GROUPS[name]
            for feature in FEATURE_GROUPS[group_name]
        )
    )
    available = set(available_columns)
    missing = [feature for feature in requested if feature not in available]
    if strict and missing:
        raise ValueError(f"Feature set {name!r} is missing columns: {missing}")
    return [feature for feature in requested if feature in available]
