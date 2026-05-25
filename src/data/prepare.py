"""src/data/prepare.py

Data preparation for cryptocurrency time series datasets.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    max_lag,
    sequence_length,
    top_k,
    train_ratio,
    valid_ratio,
    window_size,
)
from const import (
    ADJ_CLOSE_COLUMN,
    CLOSE_COLUMN,
    CRYPTO_COLUMN,
    DATA_PREP_WORKERS,
    DATE_COLUMN,
    HIGH_COLUMN,
    LOW_COLUMN,
    NUMERIC_COLUMNS,
    OPEN_COLUMN,
    VOLUME_COLUMN,
)

from .utils.adjacency_matrix import (
    build_adjacency_matrix,
)


def prepare_data(file_paths: list) -> dict:
    """Perform full data preparation pipeline.

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
            [
                df[HIGH_COLUMN],
                df[OPEN_COLUMN],
                df[CLOSE_COLUMN],
                df[LOW_COLUMN],
            ],
        )
        df[LOW_COLUMN] = np.minimum.reduce(
            [
                df[HIGH_COLUMN],
                df[OPEN_COLUMN],
                df[CLOSE_COLUMN],
                df[LOW_COLUMN],
            ],
        )

        # Add crypto identifier
        df[CRYPTO_COLUMN] = name

        cleaning_stats[name] = {
            "num_rows_before": len(df_before_cleaning),
            "num_rows_after": len(df),
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
    # TRAIN-VALID-TEST SPLITTING
    # ===================================
    train_parts = []
    valid_parts = []
    test_parts = []
    split_stats = {}

    for crypto in full_df[CRYPTO_COLUMN].unique():
        crypto_df = full_df[full_df[CRYPTO_COLUMN] == crypto].sort_values(
            DATE_COLUMN,
        )

        n = len(crypto_df)

        train_end = int(n * train_ratio)
        valid_end = int(n * (train_ratio + valid_ratio))

        train_parts.append(crypto_df.iloc[:train_end])
        valid_parts.append(crypto_df.iloc[train_end:valid_end])
        test_parts.append(crypto_df.iloc[valid_end:])

        split_stats[crypto] = {
            "num_rows": n,
            "num_train_rows": int(train_end),
            "num_valid_rows": int(valid_end - train_end),
            "num_test_rows": int(n - valid_end),
        }

    train_df = pd.concat(train_parts, axis=0)
    valid_df = pd.concat(valid_parts, axis=0)
    test_df = pd.concat(test_parts, axis=0)

    train_df = train_df.sort_values([CRYPTO_COLUMN, DATE_COLUMN])
    valid_df = valid_df.sort_values([CRYPTO_COLUMN, DATE_COLUMN])
    test_df = test_df.sort_values([CRYPTO_COLUMN, DATE_COLUMN])

    report["train_valid_test_split_statistics"] = {
        "per_crypto": split_stats,
        "total": {
            "num_train_rows": len(train_df),
            "num_valid_rows": len(valid_df),
            "num_test_rows": len(test_df),
            "train_ratio": float(train_ratio),
            "valid_ratio": float(valid_ratio),
            "test_ratio": float(1.0 - train_ratio - valid_ratio),
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
    def _apply_transform(df):
        df = df.copy()

        for col in volume_cols:
            df[col] = np.log1p(df[col])

        return df

    train_df = _apply_transform(train_df)
    valid_df = _apply_transform(valid_df)
    test_df = _apply_transform(test_df)

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
        col
        for col in full_df.columns
        if col not in [DATE_COLUMN, CRYPTO_COLUMN]
    ]

    # Compute per-crypto normalization statistics from training set
    per_crypto_normalization = {}
    cryptos = sorted(train_df[CRYPTO_COLUMN].unique())
    for crypto in cryptos:
        per_crypto_normalization[crypto] = {}
        crypto_df = train_df[train_df[CRYPTO_COLUMN] == crypto]
        for col in numeric_cols:
            mean = crypto_df[col].mean()
            std = crypto_df[col].std() if crypto_df[col].std() != 0 else 1.0
            per_crypto_normalization[crypto][col] = {
                "mean": float(mean),
                "std": float(std),
            }

    # Normalize each cryptocurrency using its own statistics to prevent data leakage
    for crypto in cryptos:
        for col in numeric_cols:
            mean = per_crypto_normalization[crypto][col]["mean"]
            std = per_crypto_normalization[crypto][col]["std"]

            # Apply normalization to train, valid, and test sets
            train_mask = train_df[CRYPTO_COLUMN] == crypto
            train_df.loc[train_mask, col] = (
                train_df.loc[train_mask, col] - mean
            ) / std

            valid_mask = valid_df[CRYPTO_COLUMN] == crypto
            valid_df.loc[valid_mask, col] = (
                valid_df.loc[valid_mask, col] - mean
            ) / std

            test_mask = test_df[CRYPTO_COLUMN] == crypto
            test_df.loc[test_mask, col] = (
                test_df.loc[test_mask, col] - mean
            ) / std

    report["normalization_statistics"] = per_crypto_normalization
    report["per_crypto_normalization"] = per_crypto_normalization
    report["cryptos_order"] = cryptos

    # ===================================
    # SLIDING WINDOW CONSTRUCTION
    # ===================================
    # Create graph-structured windows for temporal and cross-crypto relationships
    def _build_sliding_windows(df):
        feature_cols = [
            col for col in NUMERIC_COLUMNS if col != ADJ_CLOSE_COLUMN
        ]
        num_features = len(feature_cols)
        cryptos = sorted(df[CRYPTO_COLUMN].unique())
        num_cryptos = len(cryptos)
        num_nodes = num_cryptos * window_size

        # Organize data by cryptocurrency
        crypto_data = {}
        for crypto in cryptos:
            crypto_df = df[df[CRYPTO_COLUMN] == crypto].sort_values(
                DATE_COLUMN,
            )
            crypto_data[crypto] = crypto_df[feature_cols].values

        # Use minimum length to ensure balanced samples
        min_length = min(len(crypto_data[c]) for c in cryptos)
        num_samples = min_length - window_size - sequence_length + 1
        close_idx = feature_cols.index(CLOSE_COLUMN)

        # Worker function to build single window (X, y, A)
        def _build_window_item(i):
            """Build X, y, A for sample i."""
            X_window = np.zeros(
                (num_features, num_nodes, sequence_length),
                dtype=np.float32,
            )

            # Build feature window X
            for crypto_idx, crypto in enumerate(cryptos):
                for t in range(window_size):
                    node_idx = crypto_idx * window_size + t
                    time_idx = i + t
                    start_idx = time_idx - sequence_length + 1
                    X_window[:, node_idx, :] = crypto_data[crypto][
                        start_idx : time_idx + 1
                    ].T

            # Build target y
            y_sample = np.array(
                [
                    crypto_data[crypto][i + window_size, close_idx]
                    for crypto in cryptos
                ],
                dtype=np.float32,
            )

            # Build adjacency matrix A
            window_start_idx = i - sequence_length + 1
            window_end_idx = i + window_size + 1

            crypto_data_window = {
                crypto: crypto_data[crypto][window_start_idx:window_end_idx]
                for crypto in cryptos
            }

            A_i = build_adjacency_matrix(
                crypto_data_window,
                cryptos,
                window_size,
                max_lag=max_lag,
                top_k=top_k,
                close_col_idx=close_idx,
            )

            return X_window, y_sample, A_i

        # Parallelize using ProcessPoolExecutor
        sample_indices = range(
            sequence_length - 1,
            sequence_length - 1 + num_samples,
        )

        results = []
        with ProcessPoolExecutor(max_workers=DATA_PREP_WORKERS) as executor:
            futures = [
                executor.submit(_build_window_item, i) for i in sample_indices
            ]
            for future in tqdm(
                futures,
                desc="Building sliding windows & adjacency matrices",
                total=num_samples,
            ):
                results.append(future.result())

        # Unpack results
        X_list = [r[0] for r in results]
        y_list = [r[1] for r in results]
        A_list = [r[2] for r in results]

        X = np.stack(X_list, axis=0)
        y = np.stack(y_list, axis=0)
        A = np.stack(A_list, axis=0)

        return X, y, A

    X_train, y_train, A_train = _build_sliding_windows(train_df)
    X_valid, y_valid, A_valid = _build_sliding_windows(valid_df)
    X_test, y_test, A_test = _build_sliding_windows(test_df)

    report["sliding_window_statistics"] = {
        "window_size": int(window_size),
        "sequence_length": int(sequence_length),
        "num_cryptos": len(train_df[CRYPTO_COLUMN].unique()),
        "num_features": len(
            [col for col in NUMERIC_COLUMNS if col != ADJ_CLOSE_COLUMN],
        ),
        "num_nodes": int(len(train_df[CRYPTO_COLUMN].unique()) * window_size),
        "num_windows_train": int(X_train.shape[0]),
        "num_windows_valid": int(X_valid.shape[0]),
        "num_windows_test": int(X_test.shape[0]),
    }

    report["adjacency_matrix"] = {
        "lagged_cross_correlation": {
            "max_lag": int(max_lag),
            "top_k": int(top_k),
        },
    }

    return {
        "X_train": X_train,
        "y_train": y_train,
        "A_train": A_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "A_valid": A_valid,
        "X_test": X_test,
        "y_test": y_test,
        "A_test": A_test,
        "report": report,
    }
