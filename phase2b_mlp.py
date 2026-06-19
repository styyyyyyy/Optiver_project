import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

# ── 0. 路径配置 ────────────────────────────────────────────────
FEAT_PATH = Path("/Users/tianyueshao/Optiver_project/features_phase2.parquet")
DEVICE    = torch.device("cpu")  # M-chip Mac: use CPU
print(f"Using device: {DEVICE}")

# ── 1. 加载数据 ────────────────────────────────────────────────
print("Loading features...")
df = pd.read_parquet(FEAT_PATH)
print(f"Shape: {df.shape}")

# ── 2. 评估指标 ────────────────────────────────────────────────
def rmspe_numpy(y_true, y_pred):
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))

def rmspe_loss(y_pred, y_true):
    """Custom RMSPE loss for PyTorch — directly optimises competition metric."""
    pct_err = (y_true - y_pred) / (y_true + 1e-7)
    return torch.sqrt(torch.mean(pct_err ** 2))

# ── 3. 特征列定义 ──────────────────────────────────────────────
# 数值特征（需要归一化）
SKIP_COLS = {"stock_id", "time_id", "target", "bpv", "jump_var",
             "stock_cluster", "rv2_sq"}
NUM_FEATURES = [c for c in df.columns
                if c not in SKIP_COLS and df[c].dtype in [np.float64, np.float32, np.int64]
                and not c.endswith("_cluster")]
NUM_FEATURES = [c for c in NUM_FEATURES if c in df.columns]
TARGET_COL   = "target"

print(f"Numerical features: {len(NUM_FEATURES)}")

# ── 4. KNN++ CV（与 LightGBM 完全相同的折叠划分）──────────────
print("\nBuilding KNN++ CV folds (same as LightGBM)...")
train_pivot = df.pivot_table(
    index="time_id", columns="stock_id", values="rv_pred")
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

# ── 5. Dataset ────────────────────────────────────────────────
class VolDataset(Dataset):
    """
    PyTorch Dataset.
    Returns: (stock_id_tensor, numerical_features_tensor, target_tensor)
    stock_id goes through embedding layer separately.
    """
    def __init__(self, stock_ids, num_features, targets):
        self.stock_ids   = torch.LongTensor(stock_ids)
        self.num_features= torch.FloatTensor(num_features)
        self.targets     = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.stock_ids[idx], self.num_features[idx], self.targets[idx]

# ── 6. MLP 架构 ───────────────────────────────────────────────
class VolMLP(nn.Module):
    """
    Multi-Layer Perceptron with stock_id embedding.

    Architecture:
        stock_id → Embedding(n_stocks, emb_dim) → emb_dim features
        numerical → n_num features
        concat → [emb_dim + n_num]
               → Linear(512) → BN → ReLU → Dropout(0.3)
               → Linear(256) → BN → ReLU → Dropout(0.3)
               → Linear(128) → BN → ReLU → Dropout(0.2)
               → Linear(1)   → output (predicted RV)
    """
    def __init__(self, n_stocks, n_num, emb_dim=16):
        super().__init__()

        # Stock embedding — learns a 16-dim vector per stock
        # capturing each stock's unique volatility characteristics
        self.stock_emb = nn.Embedding(n_stocks, emb_dim)

        in_dim = emb_dim + n_num

        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Layer 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Layer 3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Output
            nn.Linear(128, 1),
        )

    def forward(self, stock_ids, num_feats):
        # stock embedding: (batch,) → (batch, emb_dim)
        emb = self.stock_emb(stock_ids)
        # concatenate with numerical features
        x   = torch.cat([emb, num_feats], dim=1)
        # pass through MLP
        out = self.net(x)
        return out.squeeze(1)   # (batch,)

# ── 7. Training helpers ───────────────────────────────────────
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for stock_ids, num_feats, targets in loader:
        stock_ids = stock_ids.to(DEVICE)
        num_feats = num_feats.to(DEVICE)
        targets   = targets.to(DEVICE)

        optimizer.zero_grad()
        preds = model(stock_ids, num_feats)
        loss  = rmspe_loss(preds, targets)
        loss.backward()

        # gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader):
    model.eval()
    all_preds  = []
    all_targets= []
    with torch.no_grad():
        for stock_ids, num_feats, targets in loader:
            stock_ids = stock_ids.to(DEVICE)
            num_feats = num_feats.to(DEVICE)
            preds = model(stock_ids, num_feats)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return rmspe_numpy(targets, np.clip(preds, 0, None))

# ── 8. Cross-Validation ───────────────────────────────────────
print("\n" + "="*60)
print("MLP Training — KNN++ CV (5 Folds)")
print("="*60)

N_STOCKS    = int(df["stock_id"].max()) + 1
oof_preds   = np.zeros(len(df))
is_oof      = np.zeros(len(df), dtype=bool)
fold_scores = []

# Training hyperparameters
BATCH_SIZE  = 4096
MAX_EPOCHS  = 200
LR          = 3e-4
PATIENCE    = 25       # early stopping patience (epochs)
EMB_DIM     = 16       # stock embedding dimension

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

    # ── Preprocessing ──────────────────────────────────────────
    # Fill missing values with training median
    train_medians = train_df[NUM_FEATURES].median()
    train_df[NUM_FEATURES] = train_df[NUM_FEATURES].fillna(train_medians)
    val_df[NUM_FEATURES]   = val_df[NUM_FEATURES].fillna(train_medians)

    # log1p transform: RV features are right-skewed
    # log1p brings them closer to normal distribution
    # which helps neural networks learn faster
    log_cols = [c for c in NUM_FEATURES if
                any(x in c for x in ["rv","bpv","spread","vol","trade_rv",
                                     "wap","amount","size_sum","price"])]
    log_cols = [c for c in log_cols if c in train_df.columns]

    for col in log_cols:
        train_df[col] = np.log1p(train_df[col].clip(0))
        val_df[col]   = np.log1p(val_df[col].clip(0))

    # StandardScaler: zero mean, unit variance
    # Neural networks need this — they're sensitive to feature scale
    sc = StandardScaler()
    X_train = sc.fit_transform(train_df[NUM_FEATURES].values)
    X_val   = sc.transform(val_df[NUM_FEATURES].values)

    y_train = train_df[TARGET_COL].values.astype(np.float32)
    y_val   = val_df[TARGET_COL].values.astype(np.float32)

    sid_train = train_df["stock_id"].values.astype(np.int64)
    sid_val   = val_df["stock_id"].values.astype(np.int64)

    # ── DataLoaders ────────────────────────────────────────────
    train_ds = VolDataset(sid_train, X_train.astype(np.float32), y_train)
    val_ds   = VolDataset(sid_val,   X_val.astype(np.float32),   y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2,
                              shuffle=False, num_workers=0)

    # ── Model, optimizer, scheduler ───────────────────────────
    model     = VolMLP(N_STOCKS, len(NUM_FEATURES), EMB_DIM).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=7, verbose=False)

    # ── Training loop ──────────────────────────────────────────
    best_val_rmspe = np.inf
    best_preds     = None
    patience_count = 0

    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_rmspe  = eval_epoch(model, val_loader)
        scheduler.step(val_rmspe)

        if val_rmspe < best_val_rmspe:
            best_val_rmspe = val_rmspe
            patience_count = 0
            # save best predictions
            model.eval()
            preds_list = []
            with torch.no_grad():
                for sid, nf, _ in val_loader:
                    p = model(sid.to(DEVICE), nf.to(DEVICE))
                    preds_list.append(p.cpu().numpy())
            best_preds = np.clip(np.concatenate(preds_list), 0, None)
        else:
            patience_count += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | train_loss: {train_loss:.5f} "
                  f"| val_RMSPE: {val_rmspe:.5f} "
                  f"| best: {best_val_rmspe:.5f} "
                  f"| lr: {optimizer.param_groups[0]['lr']:.2e}")

        if patience_count >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} "
                  f"(best val RMSPE: {best_val_rmspe:.5f})")
            break

    oof_preds[val_mask.values] = best_preds
    is_oof[val_mask.values]    = True
    fold_scores.append(best_val_rmspe)
    print(f"  → Fold {fold+1} best RMSPE: {best_val_rmspe:.5f}")

# ── 9. Results ────────────────────────────────────────────────
overall = rmspe_numpy(df[TARGET_COL].values[is_oof], oof_preds[is_oof])

print("\n" + "="*60)
print(f"Fold scores:       {[f'{s:.5f}' for s in fold_scores]}")
print(f"Mean CV RMSPE:     {np.mean(fold_scores):.5f}")
print(f"Std  CV RMSPE:     {np.std(fold_scores):.5f}")
print(f"Overall OOF RMSPE: {overall:.5f}")
print("="*60)

# ── 10. Visualisation ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.bar(range(1, N_FOLDS+1), fold_scores, color="#4A90D9", alpha=0.8)
ax1.axhline(np.mean(fold_scores), color="#E74C3C", linestyle="--",
            linewidth=1.5, label=f"Mean = {np.mean(fold_scores):.5f}")
ax1.set_title("MLP RMSPE per Fold", fontsize=13, fontweight="bold")
ax1.set_xlabel("Fold")
ax1.set_ylabel("RMSPE")
ax1.legend()

ax2 = axes[1]
y_true = df[TARGET_COL].values
idx = np.random.choice(np.where(is_oof)[0], size=min(5000, is_oof.sum()),
                       replace=False)
ax2.scatter(oof_preds[idx], y_true[idx], alpha=0.15, s=4, color="#534AB7")
mv = max(oof_preds[idx].max(), y_true[idx].max())
ax2.plot([0, mv], [0, mv], "r--", linewidth=1, label="Perfect prediction")
ax2.set_xlabel("MLP Predicted RV")
ax2.set_ylabel("True RV (target)")
ax2.set_title(f"OOF Predictions vs Target\nMean CV RMSPE = {np.mean(fold_scores):.5f}",
              fontsize=13, fontweight="bold")
ax2.legend()

plt.tight_layout()
plt.savefig("phase2b_mlp_results.png", dpi=150)
plt.show()
print("Saved: phase2b_mlp_results.png")

# ── 11. Compare with LightGBM ─────────────────────────────────
try:
    lgb_oof = pd.read_parquet(
        "/Users/tianyueshao/Optiver_project/lgb_oof_predictions_v3.parquet")
    lgb_oof = lgb_oof[lgb_oof["is_oof"]].copy()
    merged  = df[is_oof][["stock_id","time_id","target"]].copy()
    merged  = merged.merge(
        lgb_oof[["stock_id","time_id","lgb_oof_pred"]],
        on=["stock_id","time_id"], how="left")

    lgb_score = rmspe_numpy(merged["target"].values,
                            merged["lgb_oof_pred"].values)
    mlp_score = np.mean(fold_scores)

    print(f"\nModel Comparison:")
    print(f"  LightGBM OOF RMSPE: {lgb_score:.5f}")
    print(f"  MLP OOF RMSPE:      {mlp_score:.5f}")
    print(f"  Gap:                {abs(lgb_score - mlp_score):.5f}")
except Exception as e:
    print(f"\n(Could not load LightGBM OOF for comparison: {e})")

# ── 12. 保存 OOF 预测 ─────────────────────────────────────────
df["mlp_oof_pred"] = oof_preds
df["is_oof"]       = is_oof
df[["stock_id","time_id","target","mlp_oof_pred","is_oof"]].to_parquet(
    "/Users/tianyueshao/Optiver_project/mlp_oof_predictions.parquet",
    index=False)
print("Saved: mlp_oof_predictions.parquet")