"""
src/data/preparation.py

Data preparation for cryptocurrency time series datasets.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pathlib import Path
from src.config import TRAIN_RATIO, WINDOW_SIZE
from src.const import (
    NUMERIC_COLUMNS,
    DATE_COLUMN,
    CLOSE_COLUMN,
    ADJ_CLOSE_COLUMN,
    VOLUME_COLUMN,
    DATA_PATH,
    DATA_FORMAT,
    CRYPTO_COLUMN,
    HIGH_COLUMN,
    LOW_COLUMN,
    OPEN_COLUMN,
)


def prepare_data(file_paths: list):
    """
    Perform full data preparation pipeline.

    Args:
        file_paths (list): List of CSV file paths.

    Returns:
        dict: Prepared datasets (train/test)
    """

    dataframes = {}

    # ===================================
    # DATA SELECTION
    # ===================================
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df = df.copy()

        # Remove Adj Close column
        if ADJ_CLOSE_COLUMN in df.columns:
            df = df.drop(columns=[ADJ_CLOSE_COLUMN])

        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        df = df.sort_values(DATE_COLUMN)

        dataframes[Path(file_path).stem] = df

    # ===================================
    # DATA CLEANING
    # ===================================
    cleaned_dfs = []
    
    for name, df in dataframes.items():
        df = df.sort_values(DATE_COLUMN)

        # Remove duplicates
        df = df.drop_duplicates(subset=[DATE_COLUMN])

        # Forward fill
        df = df.ffill()

        # Price consistency correction
        df[HIGH_COLUMN] = np.maximum.reduce(
            [df[HIGH_COLUMN], df[OPEN_COLUMN], df[CLOSE_COLUMN], df[LOW_COLUMN]]
        )
        df[LOW_COLUMN] = np.minimum.reduce(
            [df[HIGH_COLUMN], df[OPEN_COLUMN], df[CLOSE_COLUMN], df[LOW_COLUMN]]
        )

        # Add crypto identifier
        df[CRYPTO_COLUMN] = name

        cleaned_dfs.append(df)

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

    for crypto in full_df[CRYPTO_COLUMN].unique():
        crypto_df = full_df[full_df[CRYPTO_COLUMN] == crypto].sort_values(DATE_COLUMN)

        split_idx = int(len(crypto_df) * TRAIN_RATIO)

        train_parts.append(crypto_df.iloc[:split_idx])
        test_parts.append(crypto_df.iloc[split_idx:])

    train_df = pd.concat(train_parts, axis=0)
    test_df = pd.concat(test_parts, axis=0)

    train_df = train_df.sort_values([CRYPTO_COLUMN, DATE_COLUMN])
    test_df = test_df.sort_values([CRYPTO_COLUMN, DATE_COLUMN])

    # ===================================
    # FEATURE TRANSFORMATION
    # ===================================
    volume_cols = [col for col in full_df.columns if VOLUME_COLUMN in col]

    def apply_transform(df):
        df = df.copy()

        for col in volume_cols:
            df[col] = np.log1p(df[col])

        return df

    train_df = apply_transform(train_df)
    test_df = apply_transform(test_df)

    # ===================================
    # Z-SCORE NORMALIZATION
    # ===================================
    numeric_cols = [
        col for col in full_df.columns if col not in [DATE_COLUMN, CRYPTO_COLUMN]
    ]

    for col in numeric_cols:
        mean = train_df[col].mean()
        std = train_df[col].std() if train_df[col].std() != 0 else 1.0

        train_df[col] = (train_df[col] - mean) / std
        test_df[col] = (test_df[col] - mean) / std

    # ===================================
    # SLIDING WINDOW CONSTRUCTION
    # ===================================
    def build_windows(df):
        X = []
        y = []
        A = []

        feature_cols = [col for col in NUMERIC_COLUMNS if col != ADJ_CLOSE_COLUMN]
        cryptos = sorted(df[CRYPTO_COLUMN].unique())
        num_cryptos = len(cryptos)
        num_nodes = num_cryptos * WINDOW_SIZE

        # Prepare data per crypto for efficient windowing
        crypto_data = {}
        for crypto in cryptos:
            crypto_df = df[df[CRYPTO_COLUMN] == crypto].sort_values(DATE_COLUMN)
            crypto_data[crypto] = crypto_df[feature_cols].values

        min_length = min(len(crypto_data[c]) for c in cryptos)

        for i in range(min_length - WINDOW_SIZE):
            # Concatenate windows for all cryptos at this timestep
            window_all_cryptos = []
            for crypto_idx, crypto in enumerate(cryptos):
                window = crypto_data[crypto][i : i + WINDOW_SIZE]
                window_all_cryptos.append(window)

            X_window = np.stack(window_all_cryptos, axis=0)

            X_window = X_window.transpose(2, 0, 1)
            X_window = X_window.reshape(len(feature_cols), num_nodes, WINDOW_SIZE)

            X.append(X_window)

            # Target: Close price for each crypto at next timestep
            y_sample = []
            for crypto in cryptos:
                close_idx = feature_cols.index(CLOSE_COLUMN)
                target = crypto_data[crypto][i + WINDOW_SIZE, close_idx]
                y_sample.append(target)
            y.append(y_sample)

            # Adjacency matrix for this window
            adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            for crypto_idx in range(num_cryptos):
                for t in range(WINDOW_SIZE - 1):
                    node_t = crypto_idx * WINDOW_SIZE + t
                    node_t_plus_1 = crypto_idx * WINDOW_SIZE + t + 1
                    adj[node_t, node_t_plus_1] = 1.0

            A.append(adj)

        # (num_samples, num_features, num_nodes, WINDOW_SIZE)
        X = np.stack(X, axis=0)

        # (num_samples, num_cryptos)
        y = np.array(y)

        # (num_samples, num_nodes, num_nodes)
        A = np.array(A)

        return X, y, A

    X_train, y_train, A_train = build_windows(train_df)
    X_test, y_test, A_test = build_windows(test_df)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "A_train": A_train,
        "X_test": X_test,
        "y_test": y_test,
        "A_test": A_test,
    }


if __name__ == "__main__":
    file_paths = list(DATA_PATH.glob(DATA_FORMAT))
    prepare_data(file_paths)
