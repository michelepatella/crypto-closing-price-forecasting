"""src/data/preparation.py

Data preparation for cryptocurrency time series datasets.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pathlib import Path
from src.const import (
    NUMERIC_COLUMNS,
    DATE_COLUMN,
    CLOSE_COLUMN,
    ADJ_CLOSE_COLUMN,
    VOLUME_COLUMN,
    DATA_PATH,
    DATA_FORMAT,
    WINDOW_SIZE,
    TRAIN_RATIO,
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

        # Remove the Adj Close column
        if ADJ_CLOSE_COLUMN in df.columns:
            df = df.drop(columns=[ADJ_CLOSE_COLUMN])

        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        df = df.sort_values(DATE_COLUMN)

        dataframes[Path(file_path).stem] = df

    # ===================================
    # DATA CLEANING
    # ===================================
    for name, df in dataframes.items():
        df = df.sort_values(DATE_COLUMN)

        # Missing values
        df = df.fillna(method="ffill").fillna(method="bfill")

        # Remove duplicates
        df = df.drop_duplicates(subset=[DATE_COLUMN])

        dataframes[name] = df

    # ===================================
    # DATA INTEGRATION
    # ===================================
    merged_df = None

    for name, df in dataframes.items():
        df = df.set_index(DATE_COLUMN)
        df = df.add_prefix(f"{name}_")

        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.join(df, how="inner")

    merged_df = merged_df.dropna()
    merged_df = merged_df.sort_index()

    # ===================================
    # TRAIN-TEST SPLITTING
    # ===================================
    split_idx = int(len(merged_df) * TRAIN_RATIO)

    train_df = merged_df.iloc[:split_idx].copy()
    test_df = merged_df.iloc[split_idx:].copy()

    # ===================================
    # FEATURE TRANSFORMATION
    # ===================================
    volume_cols = [col for col in merged_df.columns if VOLUME_COLUMN in col]

    def apply_transform(df):
        df = df.copy()

        # Log transform Volume
        for col in volume_cols:
            df[col] = np.log1p(df[col])

        return df

    train_df = apply_transform(train_df)
    test_df = apply_transform(test_df)

    # ===================================
    # Z-SCORE NORMALIZATION
    # ===================================
    stats = {}

    for col in train_df.columns:
        mean = train_df[col].mean()
        std = train_df[col].std() if train_df[col].std() != 0 else 1.0

        stats[col] = (mean, std)

        train_df[col] = (train_df[col] - mean) / std
        test_df[col] = (test_df[col] - mean) / std

    # ===================================
    # SLIDING WINDOW CONSTRUCTION
    # ===================================
    def build_windows(df):
        feature_per_crypto = []

        for name in dataframes.keys():
            cols = [f"{name}_{col}" for col in NUMERIC_COLUMNS]
            feature_per_crypto.append(df[cols].values)

        data = np.stack(feature_per_crypto, axis=1)

        X, y = [], []

        close_idx = NUMERIC_COLUMNS.index(CLOSE_COLUMN)

        for i in range(len(df) - WINDOW_SIZE):
            window = data[i : i + WINDOW_SIZE]
            target = data[i + WINDOW_SIZE, :, close_idx]

            X.append(window)
            y.append(target)

        return np.array(X), np.array(y)

    # ===================================
    # TRAIN/TEST SETS CONSTRUCTION
    # ===================================
    X_train, y_train = build_windows(train_df)
    X_test, y_test = build_windows(test_df)

    # ===================================
    # GRAPH STRUCTURE DEFINITION
    # ===================================
    num_cryptos = len(dataframes)
    num_features = len([col for col in NUMERIC_COLUMNS if col != ADJ_CLOSE_COLUMN])
    num_nodes = num_cryptos * WINDOW_SIZE

    # ===================================
    # RESHAPE FOR T-MTGNN
    # ===================================
    def reshape_X(X):
        X = X.reshape(X.shape[0], WINDOW_SIZE, num_cryptos, num_features)
        X = np.transpose(X, (0, 3, 2, 1))
        return X.reshape(X.shape[0], num_features, num_nodes, WINDOW_SIZE)

    X_train = reshape_X(X_train)
    X_test = reshape_X(X_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


if __name__ == "__main__":
    file_paths = list(DATA_PATH.glob(DATA_FORMAT))
    prepare_data(file_paths)
