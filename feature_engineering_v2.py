import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# ── 0. 路径配置 ────────────────────────────────────────────────
DATA_DIR = Path("/Users/tianyueshao/Optiver_project/optiver-realized-volatility-prediction")

# ── 1. 加载原始数据 ────────────────────────────────────────────
print("Loading raw data...")
book  = pd.read_parquet(DATA_DIR / "book_train.parquet")
trade = pd.read_parquet(DATA_DIR / "trade_train.parquet")
train = pd.read_csv(DATA_DIR / "train.csv")

book["stock_id"]  = book["stock_id"].astype(int)
trade["stock_id"] = trade["stock_id"].astype(int)
print(f"Book: {book.shape} | Trade: {trade.shape} | Train: {train.shape}")

# ── 2. 工具函数 ────────────────────────────────────────────────
def realized_vol(series):
    s = series.dropna()
    return np.sqrt((s ** 2).sum())

def bipower_var(series):
    r = series.dropna().values
    if len(r) < 2:
        return np.nan
    return (np.pi / 2) * np.sum(np.abs(r[:-1]) * np.abs(r[1:]))

# ── 3. WAP 计算（4种）────────────────────────────────────────
print("\nComputing WAPs and log returns...")
book = book.sort_values(["stock_id", "time_id", "seconds_in_bucket"])

book["wap1"] = (book["bid_price1"] * book["ask_size1"] +
                book["ask_price1"] * book["bid_size1"]) / \
               (book["bid_size1"] + book["ask_size1"])
book["wap2"] = (book["bid_price2"] * book["ask_size2"] +
                book["ask_price2"] * book["bid_size2"]) / \
               (book["bid_size2"] + book["ask_size2"])
book["wap3"] = (book["bid_price1"] * book["bid_size1"] +
                book["ask_price1"] * book["ask_size1"]) / \
               (book["bid_size1"] + book["ask_size1"])
book["wap4"] = (book["bid_price2"] * book["bid_size2"] +
                book["ask_price2"] * book["ask_size2"]) / \
               (book["bid_size2"] + book["ask_size2"])

for i in [1, 2, 3, 4]:
    book[f"log_ret{i}"] = (
        book.groupby(["stock_id", "time_id"])[f"wap{i}"]
        .transform(lambda x: np.log(x).diff())
    )

book["wap_balance"]   = abs(book["wap1"] - book["wap2"])
book["price_spread"]  = (book["ask_price1"] - book["bid_price1"]) / \
                        ((book["ask_price1"] + book["bid_price1"]) / 2)
book["price_spread2"] = (book["ask_price2"] - book["bid_price2"]) / \
                        ((book["ask_price2"] + book["bid_price2"]) / 2)
book["bid_spread"]    = book["bid_price1"] - book["bid_price2"]
book["ask_spread"]    = book["ask_price1"] - book["ask_price2"]
book["bid_ask_spread"]= abs(book["bid_spread"] - book["ask_spread"])
book["total_volume"]  = (book["ask_size1"] + book["ask_size2"] +
                         book["bid_size1"] + book["bid_size2"])
book["volume_imbalance"] = abs((book["ask_size1"] + book["ask_size2"]) -
                               (book["bid_size1"] + book["bid_size2"]))
book["obi1"] = (book["bid_size1"] - book["ask_size1"]) / \
               (book["bid_size1"] + book["ask_size1"])
book["depth_imbalance"] = (
    (book["bid_size1"] + book["bid_size2"] -
     book["ask_size1"] - book["ask_size2"]) /
    (book["bid_size1"] + book["bid_size2"] +
     book["ask_size1"] + book["ask_size2"])
)

# ── 4. 全 bucket 聚合 ──────────────────────────────────────────
print("Computing full-bucket aggregations...")

full_agg = (
    book.groupby(["stock_id", "time_id"])
    .agg(
        rv_pred          = ("log_ret1", realized_vol),
        rv2              = ("log_ret2", realized_vol),
        rv3              = ("log_ret3", realized_vol),
        rv4              = ("log_ret4", realized_vol),
        wap1_std         = ("wap1",          "std"),
        wap2_std         = ("wap2",          "std"),
        wap_balance_sum  = ("wap_balance",   "sum"),
        wap_balance_max  = ("wap_balance",   "max"),
        price_spread_sum = ("price_spread",  "sum"),
        price_spread_max = ("price_spread",  "max"),
        price_spread2_sum= ("price_spread2", "sum"),
        bid_spread_sum   = ("bid_spread",    "sum"),
        bid_spread_max   = ("bid_spread",    "max"),
        ask_spread_sum   = ("ask_spread",    "sum"),
        ask_spread_max   = ("ask_spread",    "max"),
        bid_ask_sp_sum   = ("bid_ask_spread","sum"),
        bid_ask_sp_max   = ("bid_ask_spread","max"),
        total_vol_sum    = ("total_volume",  "sum"),
        total_vol_max    = ("total_volume",  "max"),
        vol_imb_sum      = ("volume_imbalance","sum"),
        vol_imb_max      = ("volume_imbalance","max"),
        obi_mean         = ("obi1",          "mean"),
        obi_std          = ("obi1",          "std"),
        obi_last         = ("obi1",          "last"),
        depth_imb_mean   = ("depth_imbalance","mean"),
        wap_range        = ("wap1", lambda x: x.max() - x.min()),
        book_count       = ("seconds_in_bucket", "count"),
    )
    .reset_index()
)

# BPV
bpv_df = (
    book.groupby(["stock_id", "time_id"])["log_ret1"]
    .apply(bipower_var).reset_index(name="bpv")
)
full_agg = full_agg.merge(bpv_df, on=["stock_id", "time_id"], how="left")
full_agg["bpv_sqrt"]  = np.sqrt(full_agg["bpv"].clip(0))
full_agg["rv2_sq"]    = full_agg["rv_pred"] ** 2
full_agg["jump_var"]  = np.maximum(full_agg["rv2_sq"] - full_agg["bpv"], 0)
full_agg["jump_frac"] = np.where(
    full_agg["rv2_sq"] > 0, full_agg["jump_var"] / full_agg["rv2_sq"], 0)
full_agg.drop(columns=["rv2_sq"], inplace=True)

# ── 5. 时间窗口 RV（6个窗口）─────────────────────────────────
print("Computing time-window features...")

def window_rv(seconds_cutoff, suffix):
    sub = book[book["seconds_in_bucket"] >= seconds_cutoff]
    return (
        sub.groupby(["stock_id", "time_id"])
        .agg(
            **{f"rv1_{suffix}": ("log_ret1", realized_vol)},
            **{f"rv2_{suffix}": ("log_ret2", realized_vol)},
        )
        .reset_index()
    )

for cutoff, suffix in [(100,"last500"),(200,"last400"),(300,"last300"),
                        (400,"last200"),(500,"last100")]:
    full_agg = full_agg.merge(
        window_rv(cutoff, suffix), on=["stock_id","time_id"], how="left")

full_agg["rv_accel"] = full_agg["rv1_last100"] / (full_agg["rv_pred"] + 1e-8)

# ── 6. 截面排名 ────────────────────────────────────────────────
print("Computing cross-sectional rank features...")
full_agg["rv_cs_rank"]     = full_agg.groupby("time_id")["rv_pred"].rank(pct=True)
full_agg["spread_cs_rank"] = full_agg.groupby("time_id")["price_spread_sum"].rank(pct=True)

# ── 7. 市场整体特征 ────────────────────────────────────────────
print("Computing market-wide features...")
vol_cols = ["rv_pred","rv2","rv1_last400","rv1_last300","rv1_last200"]
vol_cols = [c for c in vol_cols if c in full_agg.columns]
time_stats = (
    full_agg.groupby("time_id")[vol_cols]
    .agg(["mean","std"]).reset_index()
)
time_stats.columns = ["time_id"] + [
    f"mkt_{c}_{s}" for c in vol_cols for s in ["mean","std"]]
full_agg = full_agg.merge(time_stats, on="time_id", how="left")

# ── 8. Trade 특징 ──────────────────────────────────────────────
print("Computing trade features...")
trade = trade.sort_values(["stock_id","time_id","seconds_in_bucket"])
trade["log_ret_trade"] = (
    trade.groupby(["stock_id","time_id"])["price"]
    .transform(lambda x: np.log(x).diff())
)
trade["amount"] = trade["price"] * trade["size"]

trade_agg = (
    trade.groupby(["stock_id","time_id"])
    .agg(
        trade_rv       = ("log_ret_trade", realized_vol),
        trade_count    = ("order_count",   "sum"),
        trade_count_max= ("order_count",   "max"),
        size_sum       = ("size",          "sum"),
        size_max       = ("size",          "max"),
        size_min       = ("size",          "min"),
        amount_sum     = ("amount",        "sum"),
        amount_max     = ("amount",        "max"),
        unique_seconds = ("seconds_in_bucket","nunique"),
    )
    .reset_index()
)
trade_agg["size_tau"]  = np.sqrt(1 / trade_agg["unique_seconds"].clip(1))
trade_agg["size_tau2"] = np.sqrt(1 / trade_agg["trade_count"].clip(1))

def trade_window(cutoff, suffix):
    sub = trade[trade["seconds_in_bucket"] >= cutoff]
    return (
        sub.groupby(["stock_id","time_id"])
        .agg(
            **{f"trade_rv_{suffix}":    ("log_ret_trade","realized_vol" if False else realized_vol)},
            **{f"trade_count_{suffix}": ("order_count","sum")},
            **{f"size_sum_{suffix}":    ("size","sum")},
            **{f"uniq_sec_{suffix}":    ("seconds_in_bucket","nunique")},
        )
        .reset_index()
    )

for cutoff, suffix in [(200,"last400"),(300,"last300"),(400,"last200")]:
    tw = trade_window(cutoff, suffix)
    trade_agg = trade_agg.merge(tw, on=["stock_id","time_id"], how="left")
    trade_agg[f"size_tau_{suffix}"]  = np.sqrt(1/trade_agg[f"uniq_sec_{suffix}"].clip(1))
    trade_agg[f"size_tau2_{suffix}"] = np.sqrt(1/trade_agg[f"trade_count_{suffix}"].clip(1))

trade_agg["size_tau2_d"] = trade_agg["size_tau2_last400"] - trade_agg["size_tau2"]

# trade 成交量加速
trade["half"] = trade.groupby(["stock_id","time_id"])["seconds_in_bucket"] \
    .transform(lambda x: (x >= x.median()).astype(int))
trade_half = (
    trade.groupby(["stock_id","time_id","half"])["size"]
    .sum().unstack("half").reset_index()
)
trade_half.columns = ["stock_id","time_id","size_first_half","size_second_half"]
trade_half["trade_vol_accel"] = (
    trade_half["size_second_half"] / (trade_half["size_first_half"] + 1e-8))
trade_agg = trade_agg.merge(
    trade_half[["stock_id","time_id","trade_vol_accel"]],
    on=["stock_id","time_id"], how="left")

# tendency
print("Computing tendency features...")
lis = []
for (sid, tid), grp in trade.groupby(["stock_id","time_id"]):
    prices = grp["price"].values
    sizes  = grp["size"].values
    if len(prices) < 2:
        lis.append({"stock_id":sid,"time_id":tid,"tendency":0,
                    "f_max":0,"f_min":0,"df_max":0,"df_min":0,
                    "price_iqr":0,"size_iqr":0})
        continue
    df_diff  = np.diff(prices)
    mean_p   = np.mean(prices)
    tendency = np.sum((df_diff/prices[1:])*100*sizes[1:])
    lis.append({
        "stock_id": sid, "time_id": tid,
        "tendency": tendency,
        "f_max":    np.sum(prices > mean_p),
        "f_min":    np.sum(prices < mean_p),
        "df_max":   np.sum(df_diff > 0),
        "df_min":   np.sum(df_diff < 0),
        "price_iqr":np.percentile(prices,75)-np.percentile(prices,25),
        "size_iqr": np.percentile(sizes, 75)-np.percentile(sizes, 25),
    })
tendency_df = pd.DataFrame(lis)
trade_agg = trade_agg.merge(tendency_df, on=["stock_id","time_id"], how="left")

# ── 9. 股票聚类 ────────────────────────────────────────────────
print("Computing stock clusters...")
train_pivot = train.pivot(index="time_id", columns="stock_id", values="target")
corr = train_pivot.corr()
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
kmeans.fit(corr.fillna(0).values)
stock_cluster = pd.DataFrame({
    "stock_id":     corr.index.astype(int),
    "stock_cluster":kmeans.labels_,
})
print(f"Clusters: {pd.Series(kmeans.labels_).value_counts().to_dict()}")

# ── 10. 聚类级别特征（最重要的改进！）────────────────────────
# 对每个聚类，在每个 time_id 计算该聚类所有股票的平均特征值
# 这给模型提供"同行业股票现在怎么样"的信息
print("Computing cluster-level features (key improvement)...")

# 先合并 cluster 标签到 full_agg
full_agg_c = full_agg.merge(stock_cluster, on="stock_id", how="left")
full_agg_c = full_agg_c.merge(
    train[["stock_id","time_id","target"]],
    on=["stock_id","time_id"], how="left")
full_agg_c = full_agg_c.merge(trade_agg, on=["stock_id","time_id"], how="left")

# 对每个聚类，计算 time_id 级别的均值
cluster_vol_cols = [
    "rv_pred", "rv1_last400", "rv1_last300", "rv1_last200",
    "total_vol_sum", "size_sum", "trade_count",
    "price_spread_sum", "bid_spread_sum", "ask_spread_sum",
]
cluster_vol_cols = [c for c in cluster_vol_cols if c in full_agg_c.columns]

cluster_features_list = []
for cid in range(7):
    cluster_mask = full_agg_c["stock_cluster"] == cid
    cluster_data = full_agg_c[cluster_mask].groupby("time_id")[cluster_vol_cols].mean()
    cluster_data.columns = [f"{col}_{cid}c1" for col in cluster_data.columns]
    cluster_features_list.append(cluster_data)

cluster_features = pd.concat(cluster_features_list, axis=1).reset_index()
print(f"Cluster features shape: {cluster_features.shape}")

# ── 11. 合并所有特征 ────────────────────────────────────────────
print("\nMerging all features...")
features = (
    full_agg
    .merge(trade_agg,       on=["stock_id","time_id"], how="left")
    .merge(stock_cluster,   on="stock_id",             how="left")
    .merge(cluster_features,on="time_id",              how="left")
    .merge(train,           on=["stock_id","time_id"], how="left")
)

print(f"Final shape: {features.shape}")

# 缺失值检查
missing = features.isnull().sum()
if missing[missing > 0].any():
    print(f"\nMissing values:\n{missing[missing > 0]}")

# ── 12. 相关性分析 ─────────────────────────────────────────────
print("\nTop 20 correlations with target:")
num_cols = features.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in ["stock_id","time_id","target"]]
corrs = features[num_cols].corrwith(features["target"]).abs().sort_values(ascending=False)
print(corrs.head(20).to_string())

# ── 13. 保存 ──────────────────────────────────────────────────
out_path = "/Users/tianyueshao/Optiver_project/features_phase2.parquet"
features.to_parquet(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Total features: {len(features.columns) - 3}")