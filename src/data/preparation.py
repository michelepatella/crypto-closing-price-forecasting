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
    RAW_DATA_PATH,
    RAW_DATA_FORMAT,
    WINDOW_SIZE
)


def prepare_data(file_paths: list) -> None:
    """
    Perform full data preparation pipeline.

    Args:
        file_paths (list): List of CSV file paths.

    Returns:
        None
    """

    dataframes = {}
    
    # ===================================
    # DATA SELECTION
    # ===================================
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df = df.copy()

        # Drop the Adj Close column
        if ADJ_CLOSE_COLUMN in df.columns:
            df = df.drop(columns=[ADJ_CLOSE_COLUMN])

        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        df = df.sort_values(DATE_COLUMN)

        dataframes[Path(file_path).stem] = df

    # ===================================
    # DATA CLEANING
    # ===================================
    for name, df in dataframes.items():

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
    merged_df = merged_df.sort_values(DATE_COLUMN)

    # ===================================
    # FEATURE TRANSFORMATION
    # ===================================
    volume_cols = [col for col in merged_df.columns if VOLUME_COLUMN in col]

    for col in volume_cols:
        merged_df[col] = np.log1p(merged_df[col])

    # ===================================
    # SCALING (Z-SCORE)
    # ===================================
    for col in merged_df.columns:
        mean = merged_df[col].mean()
        std = merged_df[col].std()

        merged_df[col] = (merged_df[col] - mean) / std

    # ===================================
    # SLIDING WINDOW CONSTRUCTION
    # ===================================
    data = merged_df.values
    num_timesteps = data.shape[0]

    X = []
    y = []

    for i in range(num_timesteps - WINDOW_SIZE):
        window = data[i:i + WINDOW_SIZE]
        target = data[i + WINDOW_SIZE]

        X.append(window)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    # ===================================
    # GRAPH NODE STRUCTURE
    # ===================================
    num_cryptos = len(dataframes)
    num_features = len(NUMERIC_COLUMNS)
    num_nodes = num_cryptos * WINDOW_SIZE

    # ===================================
    # RESHAPING FOR MODEL
    # ===================================
    # From: (num_samples, window_size, num_features)
    # To: (num_samples, num_features, num_nodes, window_size)

    X = X.reshape(X.shape[0], WINDOW_SIZE, num_cryptos, num_features)
    X = np.transpose(X, (0, 3, 2, 1))

    X = X.reshape(X.shape[0], num_features, num_nodes, WINDOW_SIZE)

    y = y.reshape(y.shape[0], num_cryptos, num_features)
    y = y[:, :, NUMERIC_COLUMNS.index(CLOSE_COLUMN)]


if __name__ == "__main__":
    file_paths = list(RAW_DATA_PATH.glob(RAW_DATA_FORMAT))
    prepare_data(file_paths)