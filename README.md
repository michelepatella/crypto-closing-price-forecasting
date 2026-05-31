<a id="readme-top"></a>

<br/>
<div align="center">
  <h1 align="center">Cryptocurrency Closing Price Forecasting</h1>
  <p align="center">
    Predicting closing price for 17 major cryptocurrencies.
  </p>
  <p align="center">
  </p>
</div>

<br>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
</p>

<br/>

## About The Project

### ML Pipeline Overview
- **Data Quality Assurance**: Built an automated validation pipeline to guarantee dataset integrity across 17 cryptocurrency time series, including schema enforcement, missing-value auditing, timestamp consistency checks, duplicate and irregular interval detection, financial rule validation, and IQR-based anomaly profiling.

- **Exploratory Data Analysis**: Performed large-scale statistical characterization of cryptocurrency markets through descriptive analytics, distributional analysis (skewness, kurtosis, quantiles), feature correlation studies, return-volatility assessment, and intraday price-range evaluation to uncover market structure and asset-specific behaviors.

- **Data Preparation & Feature Engineering**: Designed an end-to-end preprocessing pipeline including data cleaning, forward-fill imputation, log-transformed volume normalization, per-asset z-score standardization, technical indicator generation (RSI, MACD, ATR, ROC, log returns), leakage-free chronological train/validation/test splitting, sliding-window sequence generation, and construction of dynamic lagged cross-correlation graph adjacency matrices.

- **Spatio-Temporal Graph Modeling**: Trained a [T-MTGNN](https://github.com/michelepatella/tmtgnn) model that jointly models temporal dynamics and cross-asset dependencies through self-attention mechanisms and graph diffusion, optimized with AdamW, Huber loss, gradient clipping, adaptive learning-rate scheduling, and early stopping.

- **Model Evaluation & Performance Analysis**: Assessed forecasting performance on denormalized per-asset predictions using MAE, RMSE, MAPE, and Directional Accuracy, enabling comprehensive evaluation across cryptocurrencies with heterogeneous price scales and volatility regimes.


### Results Highlights

- **1.80% average MAPE** across 17 cryptocurrencies.

- **Performance distribution:**
  - **5 assets** below **1% MAPE**
  - **11 assets** below **1.5% MAPE**
  - **14 assets** below **2.2% MAPE**

- **Large-cap cryptocurrencies:**
  - ETH — **0.55% MAPE**, **$7.60 MAE**
  - BTC — **1.27% MAPE**, **$221.92 MAE**

- **Mid-cap cryptocurrencies:**
  - XMR — **0.82% MAPE**, **$1.23 MAE**
  - LTC — **1.34% MAPE**, **$0.97 MAE**

- **Small-cap cryptocurrencies:**
  - SYS — **0.75% MAPE**, **$0.0009 MAE**
  - XCP — **1.18% MAPE**, **$0.0373 MAE**

> Discover complete results of the latest release [here](https://github.com/michelepatella/crypto-closing-price-forecasting/releases/latest).

<p align="right"><a href="#readme-top">Top ↑</a></p>

## License

Distributed under the [MIT License](https://github.com/michelepatella/crypto-closing-price-forecasting/blob/main/LICENSE).

<p align="right"><a href="#readme-top">Top ↑</a></p>
