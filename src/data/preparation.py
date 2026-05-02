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

        # Ensure numeric stability
        price_cols = ["Open", "High", "Low", "Close"]
        df[price_cols] = df[price_cols].astype(float)

        # Forward fill
        df = df.ffill()

        # Price consistency correction
        high = df["High"]
        low = df["Low"]
        open_ = df["Open"]
        close = df["Close"]

        df["High"] = np.maximum.reduce([high, open_, close, low])
        df["Low"] = np.minimum.reduce([high, open_, close, low])

        # Add crypto identifier
        df["crypto"] = name

        cleaned_dfs.append(df)

    # ===================================
    # DATA INTEGRATION
    # ===================================
    full_df = pd.concat(cleaned_dfs, axis=0)
    full_df = full_df.sort_values(["crypto", DATE_COLUMN])

    # ===================================
    # TRAIN-TEST SPLITTING
    # ===================================
    split_idx = int(len(full_df) * TRAIN_RATIO)

    train_df = full_df.iloc[:split_idx].copy()
    test_df = full_df.iloc[split_idx:].copy()

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
    stats = {}

    numeric_cols = [col for col in full_df.columns
                    if col not in [DATE_COLUMN, "crypto"]]

    for col in numeric_cols:
        mean = train_df[col].mean()
        std = train_df[col].std() if train_df[col].std() != 0 else 1.0

        stats[col] = (mean, std)

        train_df[col] = (train_df[col] - mean) / std
        test_df[col] = (test_df[col] - mean) / std

    # ===================================
    # SLIDING WINDOW CONSTRUCTION
    # ===================================
    def build_windows(df):
        X = []
        y = []

        feature_cols = [col for col in NUMERIC_COLUMNS if col != ADJ_CLOSE_COLUMN]
        cryptos = df["crypto"].unique()

        for crypto in cryptos:
            crypto_df = df[df["crypto"] == crypto].sort_values(DATE_COLUMN)

            values = crypto_df[feature_cols].values

            for i in range(len(values) - WINDOW_SIZE):
                window = values[i:i + WINDOW_SIZE]
                target = values[i + WINDOW_SIZE][feature_cols.index(CLOSE_COLUMN)]

                X.append(window)
                y.append(target)

        return np.array(X), np.array(y)

    X_train, y_train = build_windows(train_df)
    X_test, y_test = build_windows(test_df)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "stats": stats
    }


if __name__ == "__main__":
    file_paths = list(DATA_PATH.glob(DATA_FORMAT))
    prepare_data(file_paths)
