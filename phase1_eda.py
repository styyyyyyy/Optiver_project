import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── 0. 路径配置 ──────────────────────────────────────────────
DATA_DIR = Path("/Users/tianyueshao/Optiver_project/optiver-realized-volatility-prediction")

# ── 1. 加载数据 ───────────────────────────────────────────────
print("Loading data...")
book  = pd.read_parquet(DATA_DIR / "book_train.parquet")
trade = pd.read_parquet(DATA_DIR / "trade_train.parquet")
train = pd.read_csv(DATA_DIR / "train.csv")

book["stock_id"]  = book["stock_id"].astype(int)
trade["stock_id"] = trade["stock_id"].astype(int)

print(f"Book:  {book.shape}  | Trade: {trade.shape}  | Train: {train.shape}")
print(book.dtypes)
print(book.head(3))

# ── 2. 核心特征：WAP & log return ─────────────────────────────
def calc_wap(df, level=1):
    bp  = df[f"bid_price{level}"]
    ap  = df[f"ask_price{level}"]
    bs  = df[f"bid_size{level}"]
    as_ = df[f"ask_size{level}"]
    return (bp * as_ + ap * bs) / (bs + as_)

book["wap1"] = calc_wap(book, 1)
book["wap2"] = calc_wap(book, 2)

# 排序后计算 log return
book = book.sort_values(["stock_id", "time_id", "seconds_in_bucket"])
book["log_ret1"] = (
    book.groupby(["stock_id", "time_id"])["wap1"]
    .transform(lambda x: np.log(x).diff())
)
book["log_ret2"] = (
    book.groupby(["stock_id", "time_id"])["wap2"]
    .transform(lambda x: np.log(x).diff())
)

# ── 3. Realized Volatility per bucket ─────────────────────────
def realized_vol(series):
    """RV = sqrt(sum of squared log returns)，跳过 NaN"""
    s = series.dropna()
    return np.sqrt((s ** 2).sum())

rv = (
    book.groupby(["stock_id", "time_id"])["log_ret1"]
    .apply(realized_vol)
    .reset_index(name="rv_pred")
)

# 合并 target
rv = rv.merge(train, on=["stock_id", "time_id"])
print(f"\nRV vs Target correlation: {rv['rv_pred'].corr(rv['target']):.4f}")

# ── 4. Order Book Imbalance (OBI) ──────────────────────────────
book["obi1"]    = (book["bid_size1"] - book["ask_size1"]) / \
                  (book["bid_size1"] + book["ask_size1"])
book["spread1"] = book["ask_price1"] - book["bid_price1"]

obi_features = (
    book.groupby(["stock_id", "time_id"])
    .agg(
        obi_mean   = ("obi1",    "mean"),
        obi_std    = ("obi1",    "std"),
        spread_mean= ("spread1", "mean"),
        spread_std = ("spread1", "std"),
    )
    .reset_index()
)

# ── 5. Bipower Variation（修复版）────────────────────────────
# 关键修复：
#   RV²  = Σ r²          （variance 量纲）
#   BPV  = π/2 * Σ|r_i||r_{i+1}|  （同为 variance 量纲）
#   Jump = max(RV² - BPV, 0)  → 再开根号得到 vol 量纲
#   Jump fraction = Jump_var / RV²  （在 variance 层面比较，避免量纲错误）

def bipower_var(series):
    """BPV = (π/2) * Σ |r_i| * |r_{i+1}|，与 RV² 同量纲"""
    r = series.dropna().values
    if len(r) < 2:
        return np.nan
    return (np.pi / 2) * np.sum(np.abs(r[:-1]) * np.abs(r[1:]))

bpv = (
    book.groupby(["stock_id", "time_id"])["log_ret1"]
    .apply(bipower_var)
    .reset_index(name="bpv")
)

rv = rv.merge(bpv, on=["stock_id", "time_id"], how="left")
rv["bpv"] = rv["bpv"].fillna(0)

# 在 variance 层面做差，再开根号
rv["rv2"]       = rv["rv_pred"] ** 2
rv["jump_var"]  = np.maximum(rv["rv2"] - rv["bpv"], 0)
rv["jump_sqrt"] = np.sqrt(rv["jump_var"])           # vol 量纲的 jump
rv["jump_frac"] = np.where(
    rv["rv2"] > 0,
    rv["jump_var"] / rv["rv2"],                     # variance 层面的占比
    0
)

print(f"\nJump fraction (mean): {rv['jump_frac'].mean():.3f}")
# 修复后应在 0.05 ~ 0.15 之间

# ── 6. Trade 侧特征（修复版）────────────────────────────────
# 修复：先排序，price_vol 单独计算保证顺序正确
trade = trade.sort_values(["stock_id", "time_id", "seconds_in_bucket"])

trade_features = (
    trade.groupby(["stock_id", "time_id"])
    .agg(
        trade_count     = ("order_count", "sum"),
        trade_size_mean = ("size",        "mean"),
        trade_size_std  = ("size",        "std"),
    )
    .reset_index()
)

# price_vol 单独算，确保 diff() 顺序正确
price_vol = (
    trade.groupby(["stock_id", "time_id"])["price"]
    .apply(lambda x: np.log(x).diff().std())
    .reset_index(name="price_vol")
)

trade_features = trade_features.merge(price_vol, on=["stock_id", "time_id"])

# ── 7. 合并全部特征 ───────────────────────────────────────────
features = (
    rv
    .merge(obi_features,   on=["stock_id", "time_id"], how="left")
    .merge(trade_features, on=["stock_id", "time_id"], how="left")
)

print(f"\nFeature table shape: {features.shape}")
print(features.describe())

# ── 8. EDA 可视化 ─────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig)

# 8a. RV 分布
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(np.log1p(rv["rv_pred"]), bins=60, color="#1D9E75", alpha=0.7, label="RV pred")
ax1.hist(np.log1p(rv["target"]),  bins=60, color="#7F77DD", alpha=0.5, label="Target")
ax1.set_title("log(1+RV) distribution")
ax1.set_xlabel("log(1+RV)")
ax1.set_ylabel("Count")
ax1.legend()

# 8b. RV pred vs target scatter
ax2 = fig.add_subplot(gs[0, 1])
sample = rv.sample(min(5000, len(rv)), random_state=42)
ax2.scatter(sample["rv_pred"], sample["target"], alpha=0.15, s=4, color="#534AB7")
ax2.set_xlabel("RV predicted")
ax2.set_ylabel("Target RV")
ax2.set_title("RV pred vs Target (sample)")

# 8c. Jump fraction distribution（修复后应集中在 0~0.3）
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(rv["jump_frac"].clip(0, 1), bins=50, color="#D85A30", alpha=0.75)
ax3.set_title("Jump fraction (variance level): (RV² − BPV) / RV²")
ax3.set_xlabel("Jump fraction")
ax3.set_ylabel("Count")

# 8d. Vol per stock
ax4 = fig.add_subplot(gs[1, 1])
stock_rv = rv.groupby("stock_id")["target"].median().sort_values()
ax4.bar(range(len(stock_rv)), stock_rv.values, color="#378ADD", width=1.0)
ax4.set_title("Median target RV by stock (sorted)")
ax4.set_xlabel("Stock rank")
ax4.set_ylabel("Median RV")

plt.tight_layout()
plt.savefig("phase1_eda.png", dpi=150)
print("\nSaved: phase1_eda.png")

# ── 9. 保存特征表 ─────────────────────────────────────────────
features.to_parquet("features_phase1.parquet", index=False)
print("Saved: features_phase1.parquet")