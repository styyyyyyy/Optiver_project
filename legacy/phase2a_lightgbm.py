import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

# ── 0. 路径配置 ────────────────────────────────────────────────
DATA_DIR = Path("/Users/tianyueshao/Optiver_project/optiver-realized-volatility-prediction")
FEAT_PATH = Path("/Users/tianyueshao/Optiver_project/features_phase1.parquet")

# ── 1. 加载数据 ────────────────────────────────────────────────
print("Loading features...")
df = pd.read_parquet(FEAT_PATH)
print(f"Shape: {df.shape}")

# ── 2. 评估指标 ────────────────────────────────────────────────
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))

def lgb_rmspe(y_pred, data):
    y_true = data.get_label()
    return "RMSPE", rmspe(y_true, y_pred), False


# ── 基础列预处理 ──────────────────────────────────────────────
df["bpv_sqrt"]  = np.sqrt(df["bpv"].clip(0))
df["jump_frac"] = df["jump_frac"].fillna(0)
# ── 3. 按 unique time_id 切分，不按行数 ────────────────
# 先获取所有唯一 time_id，排序后切折
unique_tids = np.sort(df["time_id"].unique())
n_tids      = len(unique_tids)
N_FOLDS     = 5
fold_size   = n_tids // N_FOLDS

print(f"Unique time_ids: {n_tids}, fold_size: {fold_size} time_ids per fold")

# ── 4. 基础特征（不含 stock-level，不含 target 衍生）────────────
BASE_FEATURES = [
    "rv_pred", "bpv_sqrt", "jump_frac",
    "obi_mean", "obi_std",
    "spread_mean", "spread_std",
    "trade_count", "trade_size_mean", "trade_size_std",
    "price_vol",
]

TARGET_COL = "target"

# ── 5. LightGBM 参数 ───────────────────────────────────────────
LGB_PARAMS = {
    "objective":         "regression",
    "metric":            "None",
    "boosting_type":     "gbdt",
    "num_leaves":        128,
    "learning_rate":     0.05,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      1,
    "min_child_samples": 20,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "verbose":           -1,
    "n_jobs":            -1,
    "random_state":      42,
}

# ── 6. 交叉验证 ────────────────────────────────────────────────
print("\n" + "="*60)
print("Time-Series Cross Validation (4 Folds, split by time_id)")
print("="*60)

# 修复点5：用独立的 is_oof 数组记录哪些行有预测，不用 > 0 判断
oof_preds      = np.zeros(len(df))
is_oof         = np.zeros(len(df), dtype=bool)
models         = []
fold_scores    = []
all_importance = []   # 修复点1：收集所有 fold 的 importance
feature_cols_final = None

for fold in range(N_FOLDS - 1):
    # 修复点3：按 time_id 切，保证同一个 time_id 不被切两边
    train_tids_end = fold_size * (fold + 1)
    val_tids_end   = train_tids_end + fold_size if fold < N_FOLDS - 2 else n_tids

    train_tids = unique_tids[:train_tids_end]
    val_tids   = unique_tids[train_tids_end:val_tids_end]

    train_mask = df["time_id"].isin(train_tids)
    val_mask   = df["time_id"].isin(val_tids)

    train_df = df[train_mask].copy()
    val_df   = df[val_mask].copy()

    print(f"\nFold {fold+1}/{N_FOLDS-1} | "
          f"Train time_ids: {len(train_tids)}  Val time_ids: {len(val_tids)} | "
          f"Train rows: {len(train_df):,}  Val rows: {len(val_df):,}")

    # ── 修复点4a：缺失值用训练集 median 填，不用全量 ──────────────
    train_medians = train_df[BASE_FEATURES].median()
    train_df[BASE_FEATURES] = train_df[BASE_FEATURES].fillna(train_medians)
    val_df[BASE_FEATURES]   = val_df[BASE_FEATURES].fillna(train_medians)

    # ── 修复点2：stock-level 特征用 expanding mean 风格 ───────────
    # 对训练集：用除自身外的历史均值（leave-one-out expanding mean）
    # 简化实现：按 time_id 顺序 expanding，每行只看之前 time_id 的数据
    #
    # 具体做法：
    #   1. 计算每个 stock 在训练集里的整体均值（不含当前行 time_id）
    #   2. 对验证集：用整个训练集的均值（安全，无泄露）

    # 训练集 stock stats（整体，用于验证集）
    stock_stats_for_val = (
        train_df.groupby("stock_id")[TARGET_COL]
        .agg(stock_rv_mean="mean", stock_rv_median="median", stock_rv_std="std")
        .reset_index()
    )

    # 训练集内 expanding mean：每个 time_id 只用之前 time_id 的均值
    # 按 time_id 排序后，逐 time_id 累积
    train_df = train_df.sort_values("time_id")
    stock_expanding = {}   # stock_id → list of (time_id, target)

    expand_mean   = np.full(len(train_df), np.nan)
    expand_median = np.full(len(train_df), np.nan)
    expand_std    = np.full(len(train_df), np.nan)

    # 按 time_id 分组处理
    global_mean = train_df[TARGET_COL].mean()
    for tid in np.sort(train_df["time_id"].unique()):
        tid_mask = train_df["time_id"] == tid
        for idx in train_df.index[tid_mask]:
            sid = train_df.at[idx, "stock_id"]
            hist = stock_expanding.get(sid, [])
            if len(hist) >= 2:
                expand_mean[train_df.index.get_loc(idx)]   = np.mean(hist)
                expand_median[train_df.index.get_loc(idx)] = np.median(hist)
                expand_std[train_df.index.get_loc(idx)]    = np.std(hist)
            else:
                expand_mean[train_df.index.get_loc(idx)]   = global_mean
                expand_median[train_df.index.get_loc(idx)] = global_mean
                expand_std[train_df.index.get_loc(idx)]    = 0.0
        # 这个 time_id 结束后，把当前 time_id 的 target 加入历史
        for idx in train_df.index[tid_mask]:
            sid = train_df.at[idx, "stock_id"]
            if sid not in stock_expanding:
                stock_expanding[sid] = []
            stock_expanding[sid].append(train_df.at[idx, TARGET_COL])

    train_df = train_df.reset_index(drop=True)
    train_df["stock_rv_mean"]   = expand_mean
    train_df["stock_rv_median"] = expand_median
    train_df["stock_rv_std"]    = expand_std

    # 验证集：用整个训练集的 stock stats（安全）
    val_df = val_df.merge(stock_stats_for_val, on="stock_id", how="left")
    val_df["stock_rv_mean"]   = val_df["stock_rv_mean"].fillna(global_mean)
    val_df["stock_rv_median"] = val_df["stock_rv_median"].fillna(global_mean)
    val_df["stock_rv_std"]    = val_df["stock_rv_std"].fillna(0)

    # 相对特征
    train_df["rv_vs_stock_mean"] = train_df["rv_pred"] / (train_df["stock_rv_mean"] + 1e-8)
    val_df["rv_vs_stock_mean"]   = val_df["rv_pred"]   / (val_df["stock_rv_mean"]   + 1e-8)

    FEATURE_COLS = BASE_FEATURES + [
        "stock_rv_mean", "stock_rv_median", "stock_rv_std",
        "rv_vs_stock_mean",
    ]
    FEATURE_COLS = [c for c in FEATURE_COLS if c in train_df.columns]
    if feature_cols_final is None:
        feature_cols_final = FEATURE_COLS
        print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    X_val   = val_df[FEATURE_COLS].values
    y_val   = val_df[TARGET_COL].values

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

    model = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round = 3000,
        valid_sets      = [dval],
        feval           = lgb_rmspe,
        callbacks       = [
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    val_pred = np.clip(val_pred, 0, None)

    # 修复点5：用 val_mask 记录 OOF，不用 > 0 判断
    oof_preds[val_mask.values] = val_pred
    is_oof[val_mask.values]    = True

    score = rmspe(y_val, val_pred)
    fold_scores.append(score)
    models.append(model)

    # 修复点1：收集每个 fold 的 importance
    imp = model.feature_importance(importance_type="gain")
    all_importance.append(imp)

    print(f"  → Fold {fold+1} RMSPE: {score:.5f}  "
          f"(best iter: {model.best_iteration})")

# ── 7. 总体得分 ────────────────────────────────────────────────
overall_rmspe = rmspe(df[TARGET_COL].values[is_oof], oof_preds[is_oof])

print("\n" + "="*60)
print(f"Fold scores:       {[f'{s:.5f}' for s in fold_scores]}")
print(f"Mean CV RMSPE:     {np.mean(fold_scores):.5f}")
print(f"Std  CV RMSPE:     {np.std(fold_scores):.5f}")
print(f"Overall OOF RMSPE: {overall_rmspe:.5f}")
print("="*60)

# ── 8. 修复点1：特征重要性取所有 fold 的平均 ──────────────────
mean_importance = np.mean(all_importance, axis=0)
importance = pd.DataFrame({
    "feature":    feature_cols_final,
    "importance": mean_importance,
}).sort_values("importance", ascending=False)

print("\nFeature Importance (mean gain across all folds):")
print(importance.to_string(index=False))

# ── 9. 可视化 ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
colors_bar = ["#4A90D9" if i < 3 else "#BDC3C7"
              for i in range(len(importance))]
ax1.barh(importance["feature"][::-1],
         importance["importance"][::-1],
         color=colors_bar[::-1])
ax1.set_title("Feature Importance\n(mean gain across 4 folds)",
              fontsize=12, fontweight="bold")
ax1.set_xlabel("Mean Gain Importance")

ax2 = axes[1]
y_true_all = df[TARGET_COL].values
sample_idx = np.random.choice(
    np.where(is_oof)[0],
    size=min(5000, is_oof.sum()),
    replace=False
)
ax2.scatter(oof_preds[sample_idx], y_true_all[sample_idx],
            alpha=0.15, s=4, color="#534AB7")
max_val = max(oof_preds[sample_idx].max(), y_true_all[sample_idx].max())
ax2.plot([0, max_val], [0, max_val], "r--", linewidth=1,
         label="Perfect prediction")
ax2.set_xlabel("OOF Predicted RV")
ax2.set_ylabel("True RV (target)")
ax2.set_title(
    f"OOF Predictions vs Target\nMean CV RMSPE = {np.mean(fold_scores):.5f}",
    fontsize=12, fontweight="bold")
ax2.legend()

plt.tight_layout()
plt.savefig("phase2a_results_v2.png", dpi=150)
plt.show()
print("\nSaved: phase2a_results_v2.png")

# ── 10. 保存 OOF 预测 ─────────────────────────────────────────
df["lgb_oof_pred"] = oof_preds
df["is_oof"]       = is_oof
df[["stock_id", "time_id", "target", "lgb_oof_pred", "is_oof"]].to_parquet(
    "/Users/tianyueshao/Optiver_project/lgb_oof_predictions.parquet",
    index=False
)
print("Saved: lgb_oof_predictions.parquet")
