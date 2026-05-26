"""src/data/explore.py

Data exploration for cryptocurrency time series datasets.
"""

import pandas as pd

from const import (
    CLOSE_COLUMN,
    DATE_COLUMN,
    HIGH_COLUMN,
    LOW_COLUMN,
    RAW_NUMERIC_COLUMNS,
)


def explore_data(file_path: str) -> dict:
    """Perform data exploration and produce a comprehensive report.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        dict: Exploration report.
    """
    df = pd.read_csv(file_path)
    df = df.copy()

    report = {}

    # ===================================
    # BASIC STATISTICS
    # ===================================
    stats_report = {}
    for col in RAW_NUMERIC_COLUMNS:
        stats_report[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std()),
            "var": float(df[col].var()),
        }

    report["basic_statistics"] = stats_report

    # ===================================
    # DISTRIBUTION SHAPE
    # ===================================
    dist_report = {}
    for col in RAW_NUMERIC_COLUMNS:
        dist_report[col] = {
            "skewness": float(df[col].skew()),
            "kurtosis": float(df[col].kurt()),
        }

    report["distribution_shape"] = dist_report

    # ===================================
    # QUANTILES
    # ===================================
    quantile_report = {}
    for col in RAW_NUMERIC_COLUMNS:
        quantile_report[col] = {
            "q1": float(df[col].quantile(0.25)),
            "q2_median": float(df[col].quantile(0.50)),
            "q3": float(df[col].quantile(0.75)),
        }

    report["quantiles"] = quantile_report

    # ===================================
    # CORRELATION ANALYSIS
    # ===================================
    correlation_matrix = df[RAW_NUMERIC_COLUMNS].corr()

    report["correlation_matrix"] = correlation_matrix.to_dict()

    # ===================================
    # VOLATILITY ANALYSIS
    # ===================================
    df = df.sort_values(DATE_COLUMN)
    returns = df[CLOSE_COLUMN].pct_change().dropna()

    report["returns_statistics"] = {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
    }

    # ===================================
    # PRICE RANGE ANALYSIS
    # ===================================
    price_range = df[HIGH_COLUMN] - df[LOW_COLUMN]

    report["price_range_statistics"] = {
        "mean_range": float(price_range.mean()),
        "std_range": float(price_range.std()),
        "min_range": float(price_range.min()),
        "max_range": float(price_range.max()),
    }

    return report
