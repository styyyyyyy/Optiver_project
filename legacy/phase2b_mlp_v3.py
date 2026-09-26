"""
Phase 2B v3: MLP trained on a DIFFERENT feature subset than LightGBM.

Goal: reduce error correlation between LightGBM and MLP by having them
look at different signals. LightGBM sees all 156 features. This MLP sees
a smaller, deliberately-different 60-feature subset.

Strategy for picking features:
  1. Keep the strongest individual signals (rv_pred + bpv_sqrt)
  2. SKIP the most redundant RV window features (e.g., rv1_last400 that's
     practically a copy of rv_pred for LightGBM)
  3. KEEP trade-side features LightGBM didn't emphasise
  4. KEEP microstructure features (spread, depth, tendency)
  5. SKIP most cluster features (they were strong for LGB but dominate
     decisions — MLP should see different signals)
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

FEAT_PATH = Path("/Users/tianyueshao/Optiver_project/features_phase2.parquet")
DEVICE    = torch.device("cpu")
print(f"Using device: {DEVICE}")

# ── 1. 加载数据 ────────────────────────────────────────────────
print("Loading features...")
df = pd.read_parquet(FEAT_PATH)
print(f"Shape: {df.shape}")

# ── 2. 评估指标 ────────────────────────────────────────────────
def rmspe_numpy(y_true, y_pred):
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))

def rmspe_loss(y_pred, y_true):
    pct_err = (y_true - y_pred) / (y_true + 1e-7)
    return torch.sqrt(torch.mean(pct_err ** 2))

# ── 3. 精选特征子集（与 LightGBM 形成差异）──────────────────────
# 核心思路：让 MLP 主要看 LightGBM 没那么强调的角度
CURATED_FEATURES = [
    # 核心波动率 — 必须保留一两个，否则模型没法训练
    "rv_pred",           # 主信号，必须保留
    "bpv_sqrt",          # 第二信号

    # 只保留一个时间窗口 —— 而不是像 LGB 用 6 个
    # LGB 重度依赖 rv1_last400/rv1_last500，我们让 MLP 看中间窗口
    "rv1_last300",

    # 波动率加速度 — LGB 没怎么用这个
    "rv_accel",

    # 微结构特征 — LGB importance 较低，MLP 可能能挖掘
    "obi_mean", "obi_std", "obi_last",
    "depth_imb_mean",
    "wap_range",
    "wap1_std", "wap2_std",
    "wap_balance_sum", "wap_balance_max",

    # 价差特征
    "price_spread_sum", "price_spread2_sum",
    "bid_spread_sum", "ask_spread_sum",
    "bid_ask_sp_sum", "bid_ask_sp_max",

    # 深度/成交量
    "total_vol_sum", "vol_imb_sum",

    # 数据密度
    "book_count",

    # 成交特征 — MLP 擅长连续特征，trade 侧重要
    "trade_rv",
    "trade_count", "trade_count_max",
    "size_sum", "size_max", "amount_sum",
    "unique_seconds",

    # size_tau（Kyle model 微结构）— LGB 几乎没用
    "size_tau", "size_tau2", "size_tau2_d",

    # 趋势 / 方向性 — LGB 用得少
    "trade_vol_accel",
    "tendency", "df_max", "df_min",
    "price_iqr", "size_iqr",
    "f_max", "f_min",

    # 截面排名
    "rv_cs_rank", "spread_cs_rank",

    # 只保留几个市场整体特征
    "mkt_rv_pred_mean", "mkt_rv_pred_std",

    # 股票聚类 —（用作 embedding 的辅助）
    "stock_cluster",
]

# 只保留实际存在的列
CURATED_FEATURES = [c for c in CURATED_FEATURES if c in df.columns]

# 不使用 cluster 级别的 _Xc1 特征（LGB 用了很多，MLP 故意不看）
NUM_FEATURES = CURATED_FEATURES.copy()
TARGET_COL   = "target"

print(f"\nCurated feature count: {len(NUM_FEATURES)}")
print(f"Features: {NUM_FEATURES[:10]} ... (total {len(NUM_FEATURES)})")
print(f"\nLGB used: ~156 features including 70 cluster features")
print(f"MLP uses: {len(NUM_FEATURES)} features, NO cluster-level aggregates")
print(f"→ This difference forces the models to learn from different signals")

# ── 4. KNN++ CV（保持与 LightGBM 相同折叠）─────────────────────
print("\nBuilding KNN++ CV folds...")
train_pivot = df.pivot_table(index="time_id", columns="stock_id", values="rv_pred")
train_pivot = train_pivot.fillna(train_pivot.mean())

N_FOLDS = 5
scaler_knn = MinMaxScaler(feature_range=(-1, 1))
mat = scaler_knn.fit_transform(train_pivot.values)
nind = int(mat.shape[0] / N_FOLDS)
mat  = np.c_[mat, np.arange(mat.shape[0])]

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
        j  = 0; kn = 0
        for val in f:
            j += val
            if j > luck[cycle]: break
            kn += 1
        lineNumber[cycle] = kn
        for n_iter in range(N_FOLDS):
            totDist[n_iter] = np.delete(totDist[n_iter], kn, axis=0)
        s[cycle] = mat[kn, :]
        values[cycle].append(int(mat[kn, -1]))
        mat = np.delete(mat, kn, axis=0)

fold_time_ids = [train_pivot.index[v] for v in values]
print(f"Fold sizes: {[len(f) for f in fold_time_ids]}")

# ── 5. Dataset & Model ────────────────────────────────────────
class VolDataset(Dataset):
    def __init__(self, stock_ids, num_features, targets):
        self.stock_ids    = torch.LongTensor(stock_ids)
        self.num_features = torch.FloatTensor(num_features)
        self.targets      = torch.FloatTensor(targets)
    def __len__(self): return len(self.targets)
    def __getitem__(self, idx):
        return self.stock_ids[idx], self.num_features[idx], self.targets[idx]

class VolMLPv3(nn.Module):
    """
    Slightly different architecture from v2:
    - Larger embedding (32 vs 16) — more capacity for stock-specific info
    - One extra hidden layer, but narrower
    - Different dropout schedule
    """
    def __init__(self, n_stocks, n_num, emb_dim=32):
        super().__init__()
        self.stock_emb = nn.Embedding(n_stocks, emb_dim)
        in_dim = emb_dim + n_num
        self.net = nn.Sequential(
            nn.Linear(in_dim, 384),
            nn.BatchNorm1d(384), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(384, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(64, 1),
        )
    def forward(self, stock_ids, num_feats):
        emb = self.stock_emb(stock_ids)
        x   = torch.cat([emb, num_feats], dim=1)
        return self.net(x).squeeze(1)

# ── 6. Training helpers ───────────────────────────────────────
def train_epoch(model, loader, optimizer):
    model.train()
    total = 0
    for sid, nf, y in loader:
        sid, nf, y = sid.to(DEVICE), nf.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        p = model(sid, nf)
        loss = rmspe_loss(p, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)

def eval_epoch(model, loader):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for sid, nf, y in loader:
            sid, nf = sid.to(DEVICE), nf.to(DEVICE)
            preds.append(model(sid, nf).cpu().numpy())
            tgts.append(y.numpy())
    p = np.concatenate(preds)
    t = np.concatenate(tgts)
    return rmspe_numpy(t, np.clip(p, 0, None))

# ── 7. CV Loop ────────────────────────────────────────────────
print("\n" + "="*60)
print("MLP v3 Training (curated feature subset)")
print("="*60)

N_STOCKS    = int(df["stock_id"].max()) + 1
oof_preds   = np.zeros(len(df))
is_oof      = np.zeros(len(df), dtype=bool)
fold_scores = []

BATCH_SIZE  = 4096
MAX_EPOCHS  = 200
LR          = 3e-4
PATIENCE    = 25
EMB_DIM     = 32

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

    # 填充缺失值
    train_medians = train_df[NUM_FEATURES].median()
    train_df[NUM_FEATURES] = train_df[NUM_FEATURES].fillna(train_medians)
    val_df[NUM_FEATURES]   = val_df[NUM_FEATURES].fillna(train_medians)

    # log1p for skewed features
    log_cols = [c for c in NUM_FEATURES if
                any(x in c for x in ["rv","bpv","spread","vol","trade_rv",
                                     "wap","amount","size_sum","price"])
                and "cs_rank" not in c and "accel" not in c]
    log_cols = [c for c in log_cols if c in train_df.columns]

    for col in log_cols:
        train_df[col] = np.log1p(train_df[col].clip(0))
        val_df[col]   = np.log1p(val_df[col].clip(0))

    # StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(train_df[NUM_FEATURES].values)
    X_val   = sc.transform(val_df[NUM_FEATURES].values)

    y_train = train_df[TARGET_COL].values.astype(np.float32)
    y_val   = val_df[TARGET_COL].values.astype(np.float32)

    sid_train = train_df["stock_id"].values.astype(np.int64)
    sid_val   = val_df["stock_id"].values.astype(np.int64)

    train_ds = VolDataset(sid_train, X_train.astype(np.float32), y_train)
    val_ds   = VolDataset(sid_val,   X_val.astype(np.float32),   y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE*2,
                              shuffle=False, num_workers=0)

    model     = VolMLPv3(N_STOCKS, len(NUM_FEATURES), EMB_DIM).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=7, verbose=False)

    best_val  = np.inf
    best_preds = None
    patience_count = 0

    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_rmspe  = eval_epoch(model, val_loader)
        scheduler.step(val_rmspe)

        if val_rmspe < best_val:
            best_val = val_rmspe
            patience_count = 0
            model.eval()
            pl = []
            with torch.no_grad():
                for sid, nf, _ in val_loader:
                    p = model(sid.to(DEVICE), nf.to(DEVICE))
                    pl.append(p.cpu().numpy())
            best_preds = np.clip(np.concatenate(pl), 0, None)
        else:
            patience_count += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d} | train_loss: {train_loss:.5f} "
                  f"| val_RMSPE: {val_rmspe:.5f} | best: {best_val:.5f} "
                  f"| lr: {optimizer.param_groups[0]['lr']:.2e}")

        if patience_count >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} "
                  f"(best val RMSPE: {best_val:.5f})")
            break

    oof_preds[val_mask.values] = best_preds
    is_oof[val_mask.values]    = True
    fold_scores.append(best_val)
    print(f"  → Fold {fold+1} best RMSPE: {best_val:.5f}")

# ── 8. Results ────────────────────────────────────────────────
overall = rmspe_numpy(df[TARGET_COL].values[is_oof], oof_preds[is_oof])

print("\n" + "="*60)
print(f"Fold scores:       {[f'{s:.5f}' for s in fold_scores]}")
print(f"Mean CV RMSPE:     {np.mean(fold_scores):.5f}")
print(f"Std  CV RMSPE:     {np.std(fold_scores):.5f}")
print(f"Overall OOF RMSPE: {overall:.5f}")
print("="*60)

# ── 9. 与原来的 MLP 和 LGB 对比 ─────────────────────────────────
try:
    lgb_oof = pd.read_parquet(
        "/Users/tianyueshao/Optiver_project/lgb_oof_predictions_v3.parquet")
    lgb_oof = lgb_oof[lgb_oof["is_oof"]].copy()

    mlp_old = pd.read_parquet(
        "/Users/tianyueshao/Optiver_project/mlp_oof_predictions.parquet")
    mlp_old = mlp_old[mlp_old["is_oof"]].copy()

    merged = df[is_oof][["stock_id","time_id","target"]].copy()
    merged["mlp_v3_pred"] = oof_preds[is_oof]
    merged = merged.merge(
        lgb_oof[["stock_id","time_id","lgb_oof_pred"]],
        on=["stock_id","time_id"], how="left")
    merged = merged.merge(
        mlp_old[["stock_id","time_id","mlp_oof_pred"]],
        on=["stock_id","time_id"], how="left")

    # 计算两种 error correlation
    err_lgb     = merged["target"] - merged["lgb_oof_pred"]
    err_mlp_old = merged["target"] - merged["mlp_oof_pred"]
    err_mlp_new = merged["target"] - merged["mlp_v3_pred"]

    corr_old = np.corrcoef(err_lgb, err_mlp_old)[0,1]
    corr_new = np.corrcoef(err_lgb, err_mlp_new)[0,1]

    print(f"\nError correlation with LightGBM:")
    print(f"  Original MLP (all features): {corr_old:.4f}")
    print(f"  New MLP v3 (curated):        {corr_new:.4f}")
    print(f"  Change: {corr_new - corr_old:+.4f} "
          f"(lower = more diverse = better for ensemble)")

    # Ensemble 测试
    from scipy.optimize import minimize
    def ens_score(w, a, b, y):
        w = max(0, min(1, float(w[0])))
        return rmspe_numpy(y, w*a + (1-w)*b)

    y = merged["target"].values
    a = merged["lgb_oof_pred"].values
    b_old = merged["mlp_oof_pred"].values
    b_new = merged["mlp_v3_pred"].values

    res_old = minimize(ens_score, [0.5], args=(a, b_old, y), method="Nelder-Mead")
    res_new = minimize(ens_score, [0.5], args=(a, b_new, y), method="Nelder-Mead")

    w_old = max(0, min(1, float(res_old.x[0])))
    w_new = max(0, min(1, float(res_new.x[0])))

    ens_old = rmspe_numpy(y, w_old*a + (1-w_old)*b_old)
    ens_new = rmspe_numpy(y, w_new*a + (1-w_new)*b_new)

    print(f"\nEnsemble scores:")
    print(f"  LGB + Original MLP: {ens_old:.5f}  (w={w_old:.2f}/{1-w_old:.2f})")
    print(f"  LGB + New MLP v3:   {ens_new:.5f}  (w={w_new:.2f}/{1-w_new:.2f})")
    print(f"  Improvement:        {ens_old - ens_new:+.5f}")

    # 三模型 ensemble
    print(f"\n3-model ensemble (LGB + Old MLP + New MLP):")
    best_3 = float("inf")
    for w_lgb in np.arange(0.2, 0.8, 0.05):
        for w_old_mlp in np.arange(0.0, 1-w_lgb, 0.05):
            w_new_mlp = 1 - w_lgb - w_old_mlp
            if w_new_mlp < 0: continue
            ens3 = w_lgb*a + w_old_mlp*b_old + w_new_mlp*b_new
            s = rmspe_numpy(y, ens3)
            if s < best_3:
                best_3 = s
                best_ws = (w_lgb, w_old_mlp, w_new_mlp)
    print(f"  Best 3-model RMSPE: {best_3:.5f}")
    print(f"  Best weights: LGB={best_ws[0]:.2f}, "
          f"OldMLP={best_ws[1]:.2f}, NewMLP={best_ws[2]:.2f}")

except Exception as e:
    print(f"Could not load old predictions: {e}")

# ── 10. 保存 ─────────────────────────────────────────────────
df["mlp_v3_oof_pred"] = oof_preds
df["is_oof"]          = is_oof
df[["stock_id","time_id","target","mlp_v3_oof_pred","is_oof"]].to_parquet(
    "/Users/tianyueshao/Optiver_project/mlp_v3_oof_predictions.parquet",
    index=False)
print("\nSaved: mlp_v3_oof_predictions.parquet")
