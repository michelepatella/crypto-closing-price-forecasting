from pathlib import Path

# ===================================
# DATA
# ===================================
DATE_COLUMN = "Date"
OPEN_COLUMN = "Open"
HIGH_COLUMN = "High"
LOW_COLUMN = "Low"
CLOSE_COLUMN = "Close"
ADJ_CLOSE_COLUMN = "Adj Close"
VOLUME_COLUMN = "Volume"
RSI_COLUMN = "RSI"
MACD_HIST_COLUMN = "MACD_Hist"
ATR_NORM_COLUMN = "ATR_Norm"
CRYPTO_COLUMN = "Crypto"

DATASET_COLUMNS = [
    DATE_COLUMN,
    OPEN_COLUMN,
    HIGH_COLUMN,
    LOW_COLUMN,
    CLOSE_COLUMN,
    ADJ_CLOSE_COLUMN,
    VOLUME_COLUMN,
    CRYPTO_COLUMN,
]

# Technical indicators
TECHNICAL_INDICATOR_COLUMNS = [
    RSI_COLUMN,
    MACD_HIST_COLUMN,
    ATR_NORM_COLUMN,
]

RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14

NUMERIC_COLUMNS = [
    OPEN_COLUMN,
    HIGH_COLUMN,
    LOW_COLUMN,
    CLOSE_COLUMN,
    ADJ_CLOSE_COLUMN,
    VOLUME_COLUMN,
    RSI_COLUMN,
    MACD_HIST_COLUMN,
    ATR_NORM_COLUMN,
]

TARGET_COLUMNS = [CLOSE_COLUMN]

DATA_FORMAT = "*.csv"

DATA_TIME_FREQUENCY = "1h"

DATA_PREP_WORKERS = 5

# ===================================
# PATHS
# ===================================
DATA_PATH = Path("data")
DATA_QUALITY_VERIFICATION_REPORT_PATH = Path(
    "reports/data/quality_verification/report.json",
)
DATA_EXPLORATION_REPORT_PATH = Path("reports/data/exploration/report.json")
DATA_PREPARATION_REPORT_PATH = Path("reports/data/preparation/report.json")
MODELING_TRAINING_REPORT_PATH = Path("reports/modeling/training/report.json")
MODELING_EVALUATION_REPORT_PATH = Path(
    "reports/modeling/evaluation/report.json",
)
BEST_MODEL_PATH = Path("models/best_model.pt")
