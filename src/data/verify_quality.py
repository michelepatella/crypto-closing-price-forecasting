"""src/data/verify_quality.py

Data quality verification for cryptocurrency time series datasets.
"""

import pandas as pd

from const import (
    CRYPTO_COLUMN,
    DATE_COLUMN,
    OPEN_COLUMN,
    HIGH_COLUMN,
    LOW_COLUMN,
    CLOSE_COLUMN,
    VOLUME_COLUMN,
    DATASET_COLUMNS,
    DATA_TIME_FREQUENCY,
)


def verify_data_quality(file_path: str) -> dict:
    """
    Verify data quality and produce a comprehensive report.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        dict: A comprehensive data quality report.
    """
    df = pd.read_csv(file_path)
    df = df.copy()

    report = {}

    # ===================================
    # BASIC COMPLETENESS
    # ===================================
    report["num_missing_values"] = df.isnull().sum().to_dict()
    report["missing_value_ratio"] = (df.isnull().sum() / len(df)).to_dict()

    # ===================================
    # SCHEMA CONSISTENCY
    # ===================================
    report["are_columns_valid"] = list(df.columns) == {
        col for col in DATASET_COLUMNS if col != CRYPTO_COLUMN
    }
    report["missing_columns"] = list(set(DATASET_COLUMNS) - set(df.columns))
    report["extra_columns"] = list(set(df.columns) - set(DATASET_COLUMNS))

    # ===================================
    # TIMESTAMP QUALITY
    # ===================================
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    report["num_invalid_timestamps"] = int(df[DATE_COLUMN].isnull().sum())
    report["num_duplicate_timestamps"] = int(df[DATE_COLUMN].duplicated().sum())
    report["is_time_sorted"] = bool(df[DATE_COLUMN].is_monotonic_increasing)

    # ===================================
    # TIME FREQUENCY ANALYSIS
    # ===================================
    df = df.sort_values(DATE_COLUMN)
    df_time = df.drop_duplicates(subset=[DATE_COLUMN]).dropna(subset=[DATE_COLUMN])
    time_diffs = df_time[DATE_COLUMN].diff().dropna()

    report["time_frequency"] = {
        str(k): int(v) for k, v in time_diffs.value_counts().items()
    }
    report["num_irregular_time_steps"] = int(
        (time_diffs != pd.Timedelta(DATA_TIME_FREQUENCY)).sum()
    )

    # ===================================
    # DATE RANGE ANALYSIS
    # ===================================
    report["date_range"] = {
        "start": str(df[DATE_COLUMN].min()),
        "end": str(df[DATE_COLUMN].max()),
    }

    # ===================================
    # VALUE VALIDITY
    # ===================================
    report["num_invalid_rows"] = len(
        df[
            (df[HIGH_COLUMN] < df[LOW_COLUMN])
            | (df[OPEN_COLUMN] < 0)
            | (df[CLOSE_COLUMN] < 0)
            | (df[VOLUME_COLUMN] < 0)
        ]
    )

    report["high_low_violations"] = int((df[HIGH_COLUMN] < df[LOW_COLUMN]).sum())
    report["open_high_violations"] = int((df[OPEN_COLUMN] > df[HIGH_COLUMN]).sum())
    report["open_low_violations"] = int((df[OPEN_COLUMN] < df[LOW_COLUMN]).sum())
    report["close_bounds_violations"] = int(
        (
            (df[CLOSE_COLUMN] > df[HIGH_COLUMN]) | (df[CLOSE_COLUMN] < df[LOW_COLUMN])
        ).sum()
    )

    # ===================================
    # OUTLIER DETECTION (IQR METHOD)
    # ===================================
    outlier_report = {}
    for col in [OPEN_COLUMN, HIGH_COLUMN, LOW_COLUMN, CLOSE_COLUMN, VOLUME_COLUMN]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_report[col] = int(outliers)

    report["num_outliers_iqr"] = outlier_report

    return report
