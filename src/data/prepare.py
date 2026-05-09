"""src/data/prepare.py

Data preparation for cryptocurrency time series datasets.
"""

import numpy as np
import pandas as pd

from pathlib import Path
from config import TRAIN_RATIO, WINDOW_SIZE
from const import (
    NUMERIC_COLUMNS,
    DATE_COLUMN,
    CLOSE_COLUMN,
    ADJ_CLOSE_COLUMN,
    VOLUME_COLUMN,
    CRYPTO_COLUMN,
    HIGH_COLUMN,
    LOW_COLUMN,
    OPEN_COLUMN,
)


def prepare_data(file_paths: list) -> dict:
    """
    Perform full data preparation pipeline.

    Args:
        file_paths (list): List of CSV file paths.

    Returns:
        dict: Prepared datasets (train/test) with report.
    """

    report = {}
    dataframes = {}
    missing_values_raw = {}

    # ===================================
    # DATA SELECTION
    # ===================================
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df = df.copy()

        # Track missing values before cleaning
        missing_values_raw[Path(file_path).stem] = {
            col: int(df[col].isnull().sum()) for col in df.columns
        }

        # Remove Adj Close column (redundant feature)
        if ADJ_CLOSE_COLUMN in df.columns:
            df = df.drop(columns=[ADJ_CLOSE_COLUMN])

        # Parse dates and ensure chronological order
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        df = df.sort_values(DATE_COLUMN)

        dataframes[Path(file_path).stem] = df

    report["num_missing_values_before"] = missing_values_raw

    # ===================================
    # DATA CLEANING
    # ===================================
    cleaned_dfs = []
    cleaning_stats = {}

    for name, df in dataframes.items():
        df_before_cleaning = df.copy()
        df = df.sort_values(DATE_COLUMN)

        # Remove duplicate timestamps, keep first occurrence
        df = df.drop_duplicates(subset=[DATE_COLUMN])

        # Fill missing values with last known value
        df = df.ffill()

        # Ensure HIGH >= all prices and LOW <= all prices
        df[HIGH_COLUMN] = np.maximum.reduce(
            [df[HIGH_COLUMN], df[OPEN_COLUMN], df[CLOSE_COLUMN], df[LOW_COLUMN]]
        )
        df[LOW_COLUMN] = np.minimum.reduce(
            [df[HIGH_COLUMN], df[OPEN_COLUMN], df[CLOSE_COLUMN], df[LOW_COLUMN]]
        )

        # Add crypto identifier
        df[CRYPTO_COLUMN] = name

        cleaning_stats[name] = {
            "num_rows_before": int(len(df_before_cleaning)),
            "num_rows_after": int(len(df)),
            "num_missing_values_after": {
                col: int(df[col].isnull().sum()) for col in df.columns
            },
        }

        cleaned_dfs.append(df)

    report["data_cleaning_statistics"] = cleaning_stats

    # ===================================
    # DATA INTEGRATION
    # ===================================
    full_df = pd.concat(cleaned_dfs, axis=0)
    full_df = full_df.sort_values([CRYPTO_COLUMN, DATE_COLUMN])

    # ===================================
    # TRAIN-TEST SPLITTING
    # ===================================
    train_parts = []
    test_parts = []
    split_stats = {}

    for crypto in full_df[CRYPTO_COLUMN].unique():
        crypto_df = full_df[full_df[CRYPTO_COLUMN] == crypto].sort_values(DATE_COLUMN)

        split_idx = int(len(crypto_df) * TRAIN_RATIO)

        train_parts.append(crypto_df.iloc[:split_idx])
        test_parts.append(crypto_df.iloc[split_idx:])

        split_stats[crypto] = {
            "num_rows": int(len(crypto_df)),
            "num_train_rows": int(split_idx),
            "num_test_rows": int(len(crypto_df) - split_idx),
        }

    train_df = pd.concat(train_parts, axis=0)
    test_df = pd.concat(test_parts, axis=0)

    train_df = train_df.sort_values([CRYPTO_COLUMN, DATE_COLUMN])
    test_df = test_df.sort_values([CRYPTO_COLUMN, DATE_COLUMN])

    report["train_test_split_statistics"] = {
        "per_crypto": split_stats,
        "total": {
            "num_train_rows": int(len(train_df)),
            "num_test_rows": int(len(test_df)),
            "train_ratio": float(TRAIN_RATIO),
            "test_ratio": float(1.0 - TRAIN_RATIO),
        },
    }

    # ===================================
    # FEATURE TRANSFORMATION
    # ===================================
    volume_cols = [col for col in full_df.columns if VOLUME_COLUMN in col]
    volume_stats_before = {}
    volume_stats_after = {}

    # Record volume statistics before transformation
    for col in volume_cols:
        volume_stats_before[col] = {
            "min_train": float(train_df[col].min()),
            "max_train": float(train_df[col].max()),
            "mean_train": float(train_df[col].mean()),
            "std_train": float(train_df[col].std()),
        }

    # Apply log transformation to reduce volume skewness
    def apply_transform(df):
        df = df.copy()

        for col in volume_cols:
            df[col] = np.log1p(df[col])

        return df

    train_df = apply_transform(train_df)
    test_df = apply_transform(test_df)

    for col in volume_cols:
        volume_stats_after[col] = {
            "min_train": float(train_df[col].min()),
            "max_train": float(train_df[col].max()),
            "mean_train": float(train_df[col].mean()),
            "std_train": float(train_df[col].std()),
        }

    report["volume_transformation"] = {
        "before_log1p": volume_stats_before,
        "after_log1p": volume_stats_after,
    }

    # ===================================
    # Z-SCORE NORMALIZATION
    # ===================================
    numeric_cols = [
        col for col in full_df.columns if col not in [DATE_COLUMN, CRYPTO_COLUMN]
    ]
    normalization_stats = {}

    # Normalize using training set statistics to prevent data leakage
    for col in numeric_cols:
        mean = train_df[col].mean()
        std = train_df[col].std() if train_df[col].std() != 0 else 1.0

        train_df[col] = (train_df[col] - mean) / std
        test_df[col] = (test_df[col] - mean) / std

        normalization_stats[col] = {
            "mean": float(mean),
            "std": float(std),
        }

    report["normalization_statistics"] = normalization_stats

    # ===================================
    # SLIDING WINDOW CONSTRUCTION
    # ===================================
    # Create graph-structured windows for temporal and cross-crypto relationships
    def build_sliding_windows(df):
        X = []
        y = []
        A = []

        feature_cols = [col for col in NUMERIC_COLUMNS if col != ADJ_CLOSE_COLUMN]
        num_features = len(feature_cols)
        cryptos = sorted(df[CRYPTO_COLUMN].unique())
        num_cryptos = len(cryptos)
        num_nodes = num_cryptos * WINDOW_SIZE

        # Organize data by cryptocurrency
        crypto_data = {}
        for crypto in cryptos:
            crypto_df = df[df[CRYPTO_COLUMN] == crypto].sort_values(DATE_COLUMN)
            crypto_data[crypto] = crypto_df[feature_cols].values

        # Use minimum length to ensure balanced samples
        min_length = min(len(crypto_data[c]) for c in cryptos)

        # Build adjacency matrix for temporal connections (t → t+1)
        adj_template = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for crypto_idx in range(num_cryptos):
            for t in range(WINDOW_SIZE - 1):
                node_t = crypto_idx * WINDOW_SIZE + t
                node_t_plus_1 = crypto_idx * WINDOW_SIZE + t + 1
                adj_template[node_t, node_t_plus_1] = 1.0

        num_samples = min_length - WINDOW_SIZE

        # Extract windows with corresponding next-step targets
        for i in range(num_samples):
            # Create window for all nodes
            X_window = np.zeros(
                (num_features, num_nodes, WINDOW_SIZE), dtype=np.float32
            )

            for crypto_idx, crypto in enumerate(cryptos):
                for t in range(WINDOW_SIZE):
                    node_idx = crypto_idx * WINDOW_SIZE + t
                    X_window[:, node_idx, :] = crypto_data[crypto][
                        i : i + WINDOW_SIZE, :
                    ].T

            X.append(X_window)

            # Target: next closing price for each cryptocurrency
            y_sample = []
            for crypto in cryptos:
                close_idx = feature_cols.index(CLOSE_COLUMN)
                target = crypto_data[crypto][i + WINDOW_SIZE, close_idx]
                y_sample.append(target)
            y.append(y_sample)

        X = np.stack(X, axis=0)
        y = np.array(y)

        # Replicate adjacency matrix for each sample
        A = np.tile(adj_template, (num_samples, 1, 1))

        # Expand targets to node dimension (one target per crypto across time window)
        y_expanded = np.zeros((y.shape[0], num_nodes), dtype=np.float32)
        for crypto_idx in range(num_cryptos):
            node_start = crypto_idx * WINDOW_SIZE
            node_end = node_start + WINDOW_SIZE
            y_expanded[:, node_start:node_end] = y[:, crypto_idx : crypto_idx + 1]

        y = y_expanded

        return X, y, A

    X_train, y_train, A_train = build_sliding_windows(train_df)
    X_test, y_test, A_test = build_sliding_windows(test_df)

    report["sliding_window_statistics"] = {
        "window_size": int(WINDOW_SIZE),
        "num_cryptos": int(len(train_df[CRYPTO_COLUMN].unique())),
        "num_features": int(
            len([col for col in NUMERIC_COLUMNS if col != ADJ_CLOSE_COLUMN])
        ),
        "num_nodes": int(len(train_df[CRYPTO_COLUMN].unique()) * WINDOW_SIZE),
        "num_windows_train": int(X_train.shape[0]),
        "num_windows_test": int(X_test.shape[0]),
    }

    return {
        "X_train": X_train,
        "y_train": y_train,
        "A_train": A_train,
        "X_test": X_test,
        "y_test": y_test,
        "A_test": A_test,
        "report": report,
    }
