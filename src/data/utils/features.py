"""src/data/utils/features.py

Feature engineering module for technical indicators using pandas TA.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta

from const import (
    ATR_NORM_COLUMN,
    ATR_PERIOD,
    CLOSE_COLUMN,
    HIGH_COLUMN,
    LOG_RETURNS_COLUMN,
    LOW_COLUMN,
    MACD_FAST,
    MACD_HIST_COLUMN,
    MACD_SIGNAL,
    MACD_SLOW,
    ROC_COLUMN,
    ROC_PERIOD,
    RSI_COLUMN,
    RSI_PERIOD,
)


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for the dataset.

    Adds the following columns to the dataframe:
    - RSI(14): Relative Strength Index (momentum/reversal)
    - MACD_Hist: MACD Histogram (trend direction)
    - ATR_Norm: Normalized Average True Range (relative volatility)
    - LogReturns: Logarithmic returns (price momentum)
    - ROC(7): Rate of Change over 7 periods (momentum indicator)

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.

    Returns:
        pd.DataFrame: DataFrame with added technical indicator columns.
    """
    df = df.copy()

    # ============================================
    # RSI (Relative Strength Index)
    # ============================================
    rsi = ta.rsi(df[CLOSE_COLUMN], length=RSI_PERIOD)
    df[RSI_COLUMN] = rsi

    # ============================================
    # MACD (Moving Average Convergence Divergence)
    # ============================================
    macd = ta.macd(
        df[CLOSE_COLUMN],
        fast=MACD_FAST,
        slow=MACD_SLOW,
        signal=MACD_SIGNAL,
    )

    # MACD returns a DataFrame with columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9,
    # so extract the histogram (difference between MACD and Signal line)
    if macd is not None and not macd.empty:
        # The histogram column name follows pattern: MACDh_<fast>_<slow>_<signal>
        hist_col = f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
        if hist_col in macd.columns:
            df[MACD_HIST_COLUMN] = macd[hist_col]
        else:
            # Fallback: calculate manually if column name differs
            macd_col = f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
            signal_col = f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
            if macd_col in macd.columns and signal_col in macd.columns:
                df[MACD_HIST_COLUMN] = macd[macd_col] - macd[signal_col]
            else:
                df[MACD_HIST_COLUMN] = pd.NA
    else:
        df[MACD_HIST_COLUMN] = pd.NA

    # ===================================
    # Normalized ATR (Average True Range)
    # ===================================
    atr = ta.atr(
        high=df[HIGH_COLUMN],
        low=df[LOW_COLUMN],
        close=df[CLOSE_COLUMN],
        length=ATR_PERIOD,
    )

    if atr is not None:
        # Normalize ATR by Close price to get relative volatility
        df[ATR_NORM_COLUMN] = atr / df[CLOSE_COLUMN]
    else:
        df[ATR_NORM_COLUMN] = pd.NA

    # ============================================
    # Log Returns
    # ============================================
    # Calculate logarithmic returns
    df[LOG_RETURNS_COLUMN] = np.log(
        df[CLOSE_COLUMN] / df[CLOSE_COLUMN].shift(1),
    )

    # ============================================
    # ROC (Rate of Change)
    # ============================================
    # Calculate Rate of Change
    roc = ta.roc(df[CLOSE_COLUMN], length=ROC_PERIOD)
    df[ROC_COLUMN] = roc

    return df
