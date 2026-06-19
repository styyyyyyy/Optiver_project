# Optiver Realized Volatility Prediction

A machine learning pipeline for the [Optiver Realized Volatility Prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction) Kaggle competition. The goal is to predict short-term realized volatility for hundreds of stocks using order book and trade data.

## Pipeline Overview

| Phase | Script | Description |
|-------|--------|-------------|
| 1 | `phase1_eda.py` | Exploratory data analysis — WAP, log returns, realized volatility |
| 2a | `phase2a_lightgbm.py` / `phase2a_lightgbm_v2.py` | LightGBM model with RMSPE loss |
| 2b | `phase2b_mlp.py` / `phase2b_mlp_v3.py` | MLP neural network model |
| 2c | `phase2c_ensemble.py` | Ensemble of LightGBM + MLP via Ridge regression |
| — | `feature_engineering_v2.py` | Feature engineering (WAP, order book imbalance, jump features, etc.) |

## Key Features Engineered

- **WAP (Weighted Average Price)** at levels 1 & 2
- **Log returns** and **realized volatility** per time bucket
- **Order book imbalance**, bid-ask spread
- **Jump fraction** and **BPV** (bipower variation)
- Cross-stock and temporal aggregations

## Evaluation Metric

**RMSPE** (Root Mean Squared Percentage Error):

$$\text{RMSPE} = \sqrt{\frac{1}{n} \sum \left(\frac{y_i - \hat{y}_i}{y_i}\right)^2}$$

## Requirements

```bash
pip install pandas numpy lightgbm scikit-learn scipy matplotlib
```

## Usage

Run phases in order:

```bash
python phase1_eda.py               # EDA & feature inspection
python feature_engineering_v2.py  # Build feature set
python phase2a_lightgbm_v2.py     # Train LightGBM
python phase2b_mlp_v3.py          # Train MLP
python phase2c_ensemble.py        # Ensemble & final predictions
```

> **Note:** Raw competition data (`book_train.parquet`, `trade_train.parquet`, `train.csv`) must be downloaded from Kaggle and placed in `optiver-realized-volatility-prediction/`.
