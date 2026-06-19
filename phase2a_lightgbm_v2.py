import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# ── 0. 路径配置 ────────────────────────────────────────────────
FEAT_PATH = Path("/Users/tianyueshao/Optiver_project/features_phase2.parquet")

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

# ── 3. 特征列定义 ──────────────────────────────────────────────
# 高相关、高重要性特征
BASE_FEATURES = [
    # 核心波动率
    "rv_pred","rv2","rv3","rv4","bpv_sqrt","jump_frac",

    # 时间窗口 RV（最重要）
    "rv1_last500","rv1_last400","rv1_last300","rv1_last200","rv1_last100",
    "rv2_last500","rv2_last400","rv2_last300","rv2_last200",
    "rv_accel",

    # 订单簿
    "obi_mean","obi_std","obi_last","depth_imb_mean",
    "wap_range","wap1_std","wap2_std",
    "wap_balance_sum","wap_balance_max",

    # 价差
    "price_spread_sum","price_spread_max","price_spread2_sum",
    "bid_spread_sum","bid_spread_max",
    "ask_spread_sum","ask_spread_max",
    "bid_ask_sp_sum","bid_ask_sp_max",

    # 深度
    "total_vol_sum","total_vol_max",
    "vol_imb_sum","vol_imb_max",
    "book_count",

    # 截面排名
    "rv_cs_rank","spread_cs_rank",

    # 市场整体
    "mkt_rv_pred_mean","mkt_rv_pred_std",
    "mkt_rv2_mean","mkt_rv2_std",
    "mkt_rv1_last400_mean","mkt_rv1_last400_std",
    "mkt_rv1_last300_mean","mkt_rv1_last300_std",
    "mkt_rv1_last200_mean","mkt_rv1_last200_std",

    # 成交 RV
    "trade_rv","trade_rv_last400","trade_rv_last300","trade_rv_last200",

    # 成交量
    "trade_count","size_sum","amount_sum","unique_seconds",
    "size_tau","size_tau2","size_tau2_d",

    # 成交方向
    "trade_vol_accel","tendency","df_max","df_min",
    "price_iqr","size_iqr",

    # 聚类标签
    "stock_cluster",
]

# 聚类级别特征（自动检测所有 _Xc1 结尾的列）
cluster_cols = [c for c in df.columns if c.endswith("c1")]
BASE_FEATURES += cluster_cols

# 只保留实际存在的列
BASE_FEATURES = [c for c in BASE_FEATURES if c in df.columns]
TARGET_COL = "target"
print(f"Using {len(BASE_FEATURES)} features")
print(f"  of which cluster-level features: {len(cluster_cols)}")

# ── 4. KNN++ 分层 CV ───────────────────────────────────────────
# 借鉴 notebook：按 target 的相似性分层，每折有代表性的市场状态
# 比纯顺序切分更稳定
print("\nBuilding KNN++ stratified CV folds...")

train_pivot = df.pivot_table(
    index="time_id", columns="stock_id", values="rv_pred")
train_pivot = train_pivot.fillna(train_pivot.mean())

N_FOLDS = 5
scaler  = MinMaxScaler(feature_range=(-1, 1))
mat     = scaler.fit_transform(train_pivot.values)

nind    = int(mat.shape[0] / N_FOLDS)
mat     = np.c_[mat, np.arange(mat.shape[0])]  # add index column

np.random.seed(2021)
lineNumber = np.sort(
    np.random.choice(mat.shape[0], size=N_FOLDS, replace=False))[::-1]

totDist = [np.zeros(mat.shape[0] - N_FOLDS) for _ in range(N_FOLDS)]
values  = [[lineNumber[n]] for n in range(N_FOLDS)]
s       = [mat[lineNumber[n], :] for n in range(N_FOLDS)]

for n in range(N_FOLDS):
    mat = np.delete(mat, lineNumber[n], axis=0)

for _ in range(nind - 1):
    luck = np.random.uniform(0, 1, N_FOLDS)
    for cycle in range(N_FOLDS):
        repeated = np.tile(s[cycle], (mat.shape[0], 1))
        sumDist  = np.sum((mat[:, :-1] - repeated[:, :-1]) ** 2, axis=1)
        totDist[cycle] += sumDist
        f  = totDist[cycle] / np.sum(totDist[cycle])
        j  = 0
        kn = 0
        for val in f:
            j += val
            if j > luck[cycle]:
                break
            kn += 1
        lineNumber[cycle] = kn
        for n_iter in range(N_FOLDS):
            totDist[n_iter] = np.delete(totDist[n_iter], kn, axis=0)
        s[cycle] = mat[kn, :]
        values[cycle].append(int(mat[kn, -1]))
        mat = np.delete(mat, kn, axis=0)

# 把索引转换回 time_id
fold_time_ids = [train_pivot.index[v] for v in values]
print(f"Fold sizes (time_ids): {[len(f) for f in fold_time_ids]}")

# ── 5. LightGBM 参数 ───────────────────────────────────────────
# 修复：用 'rmse' + sample_weight 替代 custom eval
# 这让树的每次 split 都直接优化 RMSPE，而不只是 MSE
LGB_PARAMS = {
    "objective":          "regression",   # 配合 sample_weight 近似 RMSPE
    "metric":             "None",
    "boosting_type":      "gbdt",
    "num_leaves":         128,
    "learning_rate":      0.05,
    "feature_fraction":   0.7,
    "bagging_fraction":   0.8,
    "bagging_freq":       1,
    "min_child_samples":  20,
    "reg_alpha":          0.5,
    "reg_lambda":         1.0,
    "verbose":            -1,
    "n_jobs":             -1,
    "random_state":       42,
}

# ── 6. 交叉验证 ────────────────────────────────────────────────
print("\n" + "="*60)
print("KNN++ Stratified Cross Validation (5 Folds)")
print("="*60)

oof_preds      = np.zeros(len(df))
is_oof         = np.zeros(len(df), dtype=bool)
models         = []
fold_scores    = []
all_importance = []
feature_cols_final = None

for fold in range(N_FOLDS):
    val_tids   = fold_time_ids[fold]
    train_tids = np.concatenate([fold_time_ids[i]
                                 for i in range(N_FOLDS) if i != fold])

    train_mask = df["time_id"].isin(train_tids)
    val_mask   = df["time_id"].isin(val_tids)

    train_df = df[train_mask].copy()
    val_df   = df[val_mask].copy()

    print(f"\nFold {fold+1}/{N_FOLDS} | "
          f"Train: {len(train_df):,}  Val: {len(val_df):,}")

    # 用训练集 median 填充缺失值
    train_medians = train_df[BASE_FEATURES].median()
    train_df[BASE_FEATURES] = train_df[BASE_FEATURES].fillna(train_medians)
    val_df[BASE_FEATURES]   = val_df[BASE_FEATURES].fillna(train_medians)

    # stock-level 特征（expanding mean，fold 内计算，无泄露）
    stock_stats_for_val = (
        train_df.groupby("stock_id")[TARGET_COL]
        .agg(stock_rv_mean="mean", stock_rv_median="median", stock_rv_std="std")
        .reset_index()
    )

    train_df = train_df.sort_values("time_id").reset_index(drop=True)
    global_mean = train_df[TARGET_COL].mean()

    stock_hist    = {}
    expand_mean   = np.full(len(train_df), global_mean)
    expand_median = np.full(len(train_df), global_mean)
    expand_std    = np.zeros(len(train_df))

    for tid in np.sort(train_df["time_id"].unique()):
        tid_idx = train_df.index[train_df["time_id"] == tid]
        for i in tid_idx:
            sid  = train_df.at[i, "stock_id"]
            hist = stock_hist.get(sid, [])
            if len(hist) >= 2:
                expand_mean[i]   = np.mean(hist)
                expand_median[i] = np.median(hist)
                expand_std[i]    = np.std(hist)
        for i in tid_idx:
            sid = train_df.at[i, "stock_id"]
            if sid not in stock_hist:
                stock_hist[sid] = []
            stock_hist[sid].append(train_df.at[i, TARGET_COL])

    train_df["stock_rv_mean"]   = expand_mean
    train_df["stock_rv_median"] = expand_median
    train_df["stock_rv_std"]    = expand_std

    val_df = val_df.merge(stock_stats_for_val, on="stock_id", how="left")
    val_df["stock_rv_mean"]   = val_df["stock_rv_mean"].fillna(global_mean)
    val_df["stock_rv_median"] = val_df["stock_rv_median"].fillna(global_mean)
    val_df["stock_rv_std"]    = val_df["stock_rv_std"].fillna(0)

    train_df["rv_vs_stock_mean"] = train_df["rv_pred"] / \
        (train_df["stock_rv_mean"] + 1e-8)
    val_df["rv_vs_stock_mean"]   = val_df["rv_pred"] / \
        (val_df["stock_rv_mean"] + 1e-8)

    FEATURE_COLS = BASE_FEATURES + [
        "stock_rv_mean","stock_rv_median","stock_rv_std","rv_vs_stock_mean"
    ]
    FEATURE_COLS = [c for c in FEATURE_COLS if c in train_df.columns]
    if feature_cols_final is None:
        feature_cols_final = FEATURE_COLS
        print(f"Total features: {len(FEATURE_COLS)}")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    X_val   = val_df[FEATURE_COLS].values
    y_val   = val_df[TARGET_COL].values

    # ── 核心修复：sample_weight = 1/y² ────────────────────────
    # 数学上：weighted MSE with w=1/y² ≡ RMSPE
    # 这让每次树 split 都直接优化 RMSPE
    train_weights = 1.0 / np.square(y_train)
    val_weights   = 1.0 / np.square(y_val)

    dtrain = lgb.Dataset(X_train, label=y_train, weight=train_weights)
    dval   = lgb.Dataset(X_val,   label=y_val,   weight=val_weights,
                         reference=dtrain)

    model = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round = 3000,
        valid_sets      = [dval],
        feval           = lgb_rmspe,
        callbacks       = [
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    val_pred = np.clip(val_pred, 0, None)

    oof_preds[val_mask.values] = val_pred
    is_oof[val_mask.values]    = True

    score = rmspe(y_val, val_pred)
    fold_scores.append(score)
    models.append(model)
    all_importance.append(
        model.feature_importance(importance_type="gain"))

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

# ── 8. 特征重要性 ──────────────────────────────────────────────
mean_importance = np.mean(all_importance, axis=0)
importance = pd.DataFrame({
    "feature":    feature_cols_final,
    "importance": mean_importance,
}).sort_values("importance", ascending=False)

print("\nFeature Importance (mean gain across all folds):")
print(importance.head(30).to_string(index=False))

# ── 9. 可视化 ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax1 = axes[0]
top_n = importance.head(20)
colors_bar = ["#4A90D9" if i < 5 else
              "#7FB3D3" if i < 10 else "#BDC3C7"
              for i in range(len(top_n))]
ax1.barh(top_n["feature"][::-1], top_n["importance"][::-1],
         color=colors_bar[::-1])
ax1.set_title("Feature Importance — Top 20\n(mean gain across 5 folds)",
              fontsize=12, fontweight="bold")
ax1.set_xlabel("Mean Gain Importance")

ax2 = axes[1]
y_true_all = df[TARGET_COL].values
sample_idx = np.random.choice(
    np.where(is_oof)[0], size=min(5000, is_oof.sum()), replace=False)
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
plt.savefig("phase2a_results_v4.png", dpi=150)
plt.show()
print("\nSaved: phase2a_results_v4.png")

# ── 10. 保存 OOF 预测 ─────────────────────────────────────────
df["lgb_oof_pred"] = oof_preds
df["is_oof"]       = is_oof
df[["stock_id","time_id","target","lgb_oof_pred","is_oof"]].to_parquet(
    "/Users/tianyueshao/Optiver_project/lgb_oof_predictions_v3.parquet",
    index=False)
print("Saved: lgb_oof_predictions_v3.parquet")