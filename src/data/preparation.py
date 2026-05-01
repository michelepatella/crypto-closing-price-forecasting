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
        
        # Remove duplicates
        df = df.drop_duplicates(subset=[DATE_COLUMN])

        # Missing values
        df = df.ffill().bfill()

        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            df[col] = df[col].astype(float)

        high = df["High"]
        low = df["Low"]
        open_ = df["Open"]
        close = df["Close"]

        df["High"] = np.maximum.reduce([high, open_, close, low])
        df["Low"] = np.minimum.reduce([high, open_, close, low])

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

        # Stabilize heavy-tailed volume distribution
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
        X = []
        y = []

        feature_cols = [col for col in NUMERIC_COLUMNS if col != ADJ_CLOSE_COLUMN]
        names = list(dataframes.keys())

        for i in range(len(df) - WINDOW_SIZE):

            sequences = []
            targets = []

            # Temporal graphs
            for name in names:

                cols = [f"{name}_{col}" for col in feature_cols]

                seq = df.iloc[i:i + WINDOW_SIZE][cols].values
                sequences.append(seq)

                close_col = f"{name}_{CLOSE_COLUMN}"
                targets.append(df.iloc[i + WINDOW_SIZE][close_col])

            X.append(np.stack(sequences, axis=0))
            y.append(np.array(targets))

        return np.array(X), np.array(y)

    # ===================================
    # TRAIN / TEST CONSTRUCTION
    # ===================================
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
