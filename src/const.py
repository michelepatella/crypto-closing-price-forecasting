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

NUMERIC_COLUMNS = [
    OPEN_COLUMN,
    HIGH_COLUMN,
    LOW_COLUMN,
    CLOSE_COLUMN,
    ADJ_CLOSE_COLUMN,
    VOLUME_COLUMN,
]

DATA_FORMAT = "*.csv"

DATA_TIME_FREQUENCY = "1h"

# ===================================
# PATHS
# ===================================
DATA_PATH = Path("data")
DATA_QUALITY_VERIFICATION_REPORT_PATH = Path(
    "reports/data/quality_verification/report.json"
)
DATA_EXPLORATION_REPORT_PATH = Path("reports/data/exploration/report.json")
DATA_PREPARATION_REPORT_PATH = Path("reports/data/preparation/report.json")
MODELING_TRAINING_REPORT_PATH = Path("reports/modeling/training/report.json")
MODEL_CHECKPOINT_DIR = Path("models/checkpoints")
