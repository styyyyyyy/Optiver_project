# Data setup

## Why the dataset is not committed

The Optiver competition download is approximately 2.73 GB and its license is
listed by Kaggle as **Subject to Competition Rules**. Access requires accepting
those rules. This public repository therefore does not redistribute the raw
files or the derived feature cache.

GitHub also rejects ordinary Git objects larger than 100 MiB; the canonical
feature cache is about 191 MiB. Git LFS would address the technical limit but
not the redistribution permission.

## Download the raw data

1. Open the [competition data page](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data).
2. Sign in and accept the competition rules.
3. Install and authenticate the Kaggle CLI.
4. Download and extract the archive locally:

```bash
mkdir -p data/raw
kaggle competitions download \
  -c optiver-realized-volatility-prediction \
  -p data/raw
unzip data/raw/optiver-realized-volatility-prediction.zip -d data/raw
```

Expected raw layout:

```text
data/raw/
├── book_train.parquet/
├── trade_train.parquet/
├── book_test.parquet/
├── trade_test.parquet/
├── train.csv
├── test.csv
└── sample_submission.csv
```

## Frozen feature cache

The formal experiments use:

```text
data/processed/features_phase2.parquet
```

Its expected size, SHA-256, row count and schema expectations are recorded in
[`feature_cache_manifest.json`](feature_cache_manifest.json). Verify it with:

```bash
python scripts/verify_feature_cache.py \
  data/processed/features_phase2.parquet
```

The historical builder is preserved as
[`legacy/feature_engineering_v2.py`](../legacy/feature_engineering_v2.py). It is
not presented as a clean end-to-end production pipeline: it uses hard-coded
paths and also writes old full-target cluster columns. The current experiment
loader explicitly rejects those columns and selects only the audited clean
feature registry. The final study did not regenerate the cache from all raw
rows, so this limitation is kept visible rather than silently rewritten.

## Files that stay local

The `.gitignore` excludes:

- `data/raw/`
- `data/processed/`
- all Parquet files
- generated `artifacts/`

Do not force-add competition data to the public repository without first
confirming redistribution permission.
