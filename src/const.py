from pathlib import Path


DATE_COLUMN = "Date"
OPEN_COLUMN = "Open"
HIGH_COLUMN = "High"
LOW_COLUMN = "Low"
CLOSE_COLUMN = "Close"
ADJ_CLOSE_COLUMN = "Adj Close"
VOLUME_COLUMN = "Volume"

DATASET_COLUMNS = [
    DATE_COLUMN,
    OPEN_COLUMN,
    HIGH_COLUMN,
    LOW_COLUMN,
    CLOSE_COLUMN,
    ADJ_CLOSE_COLUMN,
    VOLUME_COLUMN,
]

NUMERIC_COLUMNS = [
    OPEN_COLUMN,
    HIGH_COLUMN,
    LOW_COLUMN,
    CLOSE_COLUMN,
    ADJ_CLOSE_COLUMN,
    VOLUME_COLUMN,
]

RAW_DATA_PATH = Path("data/raw")
RAW_DATA_FORMAT = "*.csv"

DATA_QUALITY_VERIFICATION_REPORT_PATH = Path(
    "reports/data/quality_verification_report.json"
)
DATA_EXPLORATION_REPORT_PATH = Path("reports/data/exploration_report.json")
