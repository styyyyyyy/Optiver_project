import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, Ridge
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

# ── 0. 路径配置 ────────────────────────────────────────────────
PROJ_DIR = Path("/Users/tianyueshao/Optiver_project")
LGB_PATH = PROJ_DIR / "lgb_oof_predictions_v3.parquet"
MLP_PATH = PROJ_DIR / "mlp_oof_predictions.parquet"

# ── 1. 加载 OOF 预测 ──────────────────────────────────────────
print("Loading OOF predictions...")
lgb_df = pd.read_parquet(LGB_PATH)
mlp_df = pd.read_parquet(MLP_PATH)

print(f"LightGBM OOF shape: {lgb_df.shape}")
print(f"MLP OOF shape:      {mlp_df.shape}")

# ── 2. 合并两个模型的预测 ─────────────────────────────────────
print("\nMerging predictions on (stock_id, time_id)...")

merged = lgb_df.merge(
    mlp_df[["stock_id", "time_id", "mlp_oof_pred"]],
    on=["stock_id", "time_id"],
    how="inner"
)

# 只保留两个模型都有 OOF 预测的行
if "is_oof" in merged.columns:
    merged = merged[merged["is_oof"]].copy()

print(f"Merged shape (both models have predictions): {merged.shape}")

# ── 3. 评估指标 ────────────────────────────────────────────────
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))

y_true  = merged["target"].values
lgb_pred = np.clip(merged["lgb_oof_pred"].values, 1e-8, None)
mlp_pred = np.clip(merged["mlp_oof_pred"].values, 1e-8, None)

# 各自的基准分数
score_lgb = rmspe(y_true, lgb_pred)
score_mlp = rmspe(y_true, mlp_pred)

print(f"\nBaseline scores:")
print(f"  LightGBM RMSPE: {score_lgb:.5f}")
print(f"  MLP RMSPE:      {score_mlp:.5f}")

# ── 4. 误差相关性分析 ──────────────────────────────────────────
# 检查两个模型的误差是否真的是"独立的"
# 相关性越低，ensemble 效果越好
print("\n" + "="*60)
print("Error Correlation Analysis")
print("="*60)

lgb_err = y_true - lgb_pred
mlp_err = y_true - mlp_pred

err_corr = np.corrcoef(lgb_err, mlp_err)[0, 1]
pred_corr = np.corrcoef(lgb_pred, mlp_pred)[0, 1]

print(f"Prediction correlation: {pred_corr:.4f}")
print(f"Error correlation:      {err_corr:.4f}")
print(f"  (Lower error correlation = better ensemble potential)")

# ── 5. 策略 1: 简单加权平均（网格搜索）────────────────────────
print("\n" + "="*60)
print("Strategy 1: Weighted Average (grid search)")
print("="*60)

best_score = float("inf")
best_w     = 0.5
scores = []
weights = np.arange(0.0, 1.01, 0.05)

for w in weights:
    ensemble = w * lgb_pred + (1 - w) * mlp_pred
    s = rmspe(y_true, ensemble)
    scores.append(s)
    if s < best_score:
        best_score = s
        best_w     = w

print(f"Best weight for LightGBM: {best_w:.2f}")
print(f"Best weight for MLP:      {1-best_w:.2f}")
print(f"Best weighted avg RMSPE:  {best_score:.5f}")

# ── 6. 策略 2: 几何平均 ────────────────────────────────────────
# 对于波动率这种正数、右偏分布，几何平均有时更好
geometric = np.sqrt(lgb_pred * mlp_pred)
score_geo = rmspe(y_true, geometric)
print(f"\nGeometric mean RMSPE: {score_geo:.5f}")

# ── 7. 策略 3: 优化最优权重（scipy）────────────────────────────
print("\n" + "="*60)
print("Strategy 3: Scipy optimization (continuous weight)")
print("="*60)

def neg_score(w, lgb_p, mlp_p, y):
    w = float(w[0])
    w = max(0, min(1, w))
    ens = w * lgb_p + (1 - w) * mlp_p
    return rmspe(y, ens)

result = minimize(
    neg_score,
    x0=[0.5],
    args=(lgb_pred, mlp_pred, y_true),
    method="Nelder-Mead",
    options={"xatol": 1e-4},
)
w_opt = float(result.x[0])
w_opt = max(0, min(1, w_opt))
score_opt = neg_score([w_opt], lgb_pred, mlp_pred, y_true)
print(f"Optimal weight: LGB={w_opt:.4f}, MLP={1-w_opt:.4f}")
print(f"Optimal RMSPE: {score_opt:.5f}")

# ── 8. 策略 4: Stacking with Linear Regression ───────────────
print("\n" + "="*60)
print("Strategy 4: Linear Stacking (meta-model)")
print("="*60)

# 把两个模型的预测作为特征，训练一个元模型
# 用 KFold 避免 meta-model 看到训练集的预测
from sklearn.model_selection import KFold

X_meta = np.column_stack([lgb_pred, mlp_pred])
y_meta = y_true

kf = KFold(n_splits=5, shuffle=True, random_state=42)
meta_preds = np.zeros(len(y_meta))

for train_idx, val_idx in kf.split(X_meta):
    X_tr, X_vl = X_meta[train_idx], X_meta[val_idx]
    y_tr, y_vl = y_meta[train_idx], y_meta[val_idx]

    # Ridge with small alpha — less prone to overfitting than pure LinReg
    meta = Ridge(alpha=0.01)
    meta.fit(X_tr, y_tr)
    meta_preds[val_idx] = np.clip(meta.predict(X_vl), 1e-8, None)

score_stack_lr = rmspe(y_meta, meta_preds)
print(f"Linear stacking RMSPE: {score_stack_lr:.5f}")

# 最终 stacking model 在全量数据上训练（用于推断）
final_meta = Ridge(alpha=0.01)
final_meta.fit(X_meta, y_meta)
print(f"Linear stacking coefficients: LGB={final_meta.coef_[0]:.4f}, "
      f"MLP={final_meta.coef_[1]:.4f}, intercept={final_meta.intercept_:.6f}")

# ── 9. 策略 5: LightGBM Stacking ──────────────────────────────
# 用小的 LightGBM 模型做 meta-learner，能捕捉非线性组合
print("\n" + "="*60)
print("Strategy 5: LightGBM Stacking (non-linear meta-model)")
print("="*60)

meta_lgb_preds = np.zeros(len(y_meta))

for train_idx, val_idx in kf.split(X_meta):
    X_tr, X_vl = X_meta[train_idx], X_meta[val_idx]
    y_tr, y_vl = y_meta[train_idx], y_meta[val_idx]

    # Sample weights for RMSPE alignment
    tr_weights = 1.0 / np.square(y_tr)

    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=tr_weights)

    meta_lgb = lgb.train(
        {
            "objective":         "regression",
            "metric":            "None",
            "num_leaves":        16,       # small — only 2 features
            "learning_rate":     0.05,
            "min_child_samples": 50,
            "reg_lambda":        1.0,
            "verbose":           -1,
            "n_jobs":            -1,
            "random_state":      42,
        },
        dtrain,
        num_boost_round=500,
    )

    meta_lgb_preds[val_idx] = np.clip(meta_lgb.predict(X_vl), 1e-8, None)

score_stack_lgb = rmspe(y_meta, meta_lgb_preds)
print(f"LightGBM stacking RMSPE: {score_stack_lgb:.5f}")

# ── 10. 汇总结果 ──────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)

results = pd.DataFrame([
    ["LightGBM alone",              score_lgb],
    ["MLP alone",                   score_mlp],
    ["Weighted average (grid)",     best_score],
    ["Weighted average (scipy)",    score_opt],
    ["Geometric mean",              score_geo],
    ["Linear stacking (Ridge)",     score_stack_lr],
    ["LightGBM stacking",           score_stack_lgb],
], columns=["Method", "RMSPE"])

results = results.sort_values("RMSPE").reset_index(drop=True)
print(results.to_string(index=False))

best_method   = results.iloc[0]["Method"]
best_rmspe    = results.iloc[0]["RMSPE"]
best_improvement_pct = (score_lgb - best_rmspe) / score_lgb * 100

print(f"\nBest method:        {best_method}")
print(f"Best RMSPE:         {best_rmspe:.5f}")
print(f"Improvement vs LGB: {(score_lgb - best_rmspe):.5f} ({best_improvement_pct:.2f}%)")
print(f"Improvement vs MLP: {(score_mlp - best_rmspe):.5f}")

# ── 11. 可视化 ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) 权重敏感性
ax1 = axes[0, 0]
ax1.plot(weights, scores, color="#4A90D9", linewidth=2, marker="o", markersize=4)
ax1.axvline(best_w, color="#E74C3C", linestyle="--", linewidth=1.5,
            label=f"Optimal w={best_w:.2f}")
ax1.axhline(score_lgb, color="#27AE60", linestyle=":", linewidth=1.5,
            label=f"LGB alone={score_lgb:.4f}")
ax1.axhline(score_mlp, color="#F39C12", linestyle=":", linewidth=1.5,
            label=f"MLP alone={score_mlp:.4f}")
ax1.set_xlabel("Weight for LightGBM")
ax1.set_ylabel("Ensemble RMSPE")
ax1.set_title("Weighted Average: Weight Sensitivity",
              fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# (b) 方法比较
ax2 = axes[0, 1]
y_pos = np.arange(len(results))
colors_bar = ["#27AE60" if i == 0 else "#4A90D9" for i in range(len(results))]
ax2.barh(y_pos, results["RMSPE"], color=colors_bar, alpha=0.8)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(results["Method"], fontsize=9)
ax2.set_xlabel("RMSPE (lower is better)")
ax2.set_title("All Methods Compared", fontsize=12, fontweight="bold")
ax2.invert_yaxis()
for i, v in enumerate(results["RMSPE"]):
    ax2.text(v, i, f"  {v:.5f}", va="center", fontsize=8)

# (c) 最佳 ensemble vs 目标散点
ax3 = axes[1, 0]
if best_method == "Weighted average (grid)":
    final_pred = best_w * lgb_pred + (1 - best_w) * mlp_pred
elif best_method == "Weighted average (scipy)":
    final_pred = w_opt * lgb_pred + (1 - w_opt) * mlp_pred
elif best_method == "Geometric mean":
    final_pred = geometric
elif best_method == "Linear stacking (Ridge)":
    final_pred = meta_preds
elif best_method == "LightGBM stacking":
    final_pred = meta_lgb_preds
else:
    final_pred = lgb_pred

idx = np.random.choice(len(y_true), min(5000, len(y_true)), replace=False)
ax3.scatter(final_pred[idx], y_true[idx], alpha=0.15, s=4, color="#534AB7")
mv = max(final_pred[idx].max(), y_true[idx].max())
ax3.plot([0, mv], [0, mv], "r--", linewidth=1, label="Perfect prediction")
ax3.set_xlabel("Ensemble Predicted RV")
ax3.set_ylabel("True RV (target)")
ax3.set_title(f"Best Ensemble ({best_method})\nRMSPE = {best_rmspe:.5f}",
              fontsize=12, fontweight="bold")
ax3.legend()

# (d) 误差对比（LGB vs MLP）
ax4 = axes[1, 1]
ax4.scatter(lgb_err[idx], mlp_err[idx], alpha=0.15, s=4, color="#9B59B6")
ax4.axhline(0, color="gray", linewidth=0.5)
ax4.axvline(0, color="gray", linewidth=0.5)
ax4.set_xlabel("LightGBM Error (y_true - y_pred)")
ax4.set_ylabel("MLP Error (y_true - y_pred)")
ax4.set_title(f"Error Independence\nCorrelation = {err_corr:.4f}",
              fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("phase2c_ensemble_results.png", dpi=150)
plt.show()
print("\nSaved: phase2c_ensemble_results.png")

# ── 12. 保存最终预测 ──────────────────────────────────────────
merged["ensemble_pred"] = final_pred
merged[["stock_id", "time_id", "target",
        "lgb_oof_pred", "mlp_oof_pred", "ensemble_pred"]].to_parquet(
    PROJ_DIR / "final_ensemble_predictions.parquet",
    index=False
)
print(f"\nSaved: {PROJ_DIR}/final_ensemble_predictions.parquet")
print(f"\n{'='*60}")
print(f"PHASE 2 COMPLETE")
print(f"{'='*60}")
print(f"  Phase 1 baseline:       0.289")
print(f"  Phase 2A (LightGBM):    {score_lgb:.5f}")
print(f"  Phase 2B (MLP):         {score_mlp:.5f}")
print(f"  Phase 2C (Ensemble):    {best_rmspe:.5f}")
print(f"  Total improvement:      {(0.289 - best_rmspe):.5f} "
      f"({(0.289 - best_rmspe)/0.289*100:.1f}%)")
