import json
from pathlib import Path

from src.const import (
    DATA_EXPLORATION_REPORT_PATH,
    DATA_FORMAT,
    DATA_PATH,
    DATA_PREPARATION_REPORT_PATH,
    DATA_QUALITY_VERIFICATION_REPORT_PATH,
)
from src.data.exploration import explore_data
from src.data.preparation import prepare_data
from src.data.quality_verification import verify_data_quality


def main() -> None:
    data_paths = list(DATA_PATH.glob(DATA_FORMAT))

    ##################################################
    # [PRE] DATA QUALITY VERIFICATION AND EXPLORATION
    ##################################################
    # Data quality verification
    data_quality_verif_full_report = {}
    for data_path in data_paths:
        data_quality_verif_report = verify_data_quality(str(data_path))
        data_quality_verif_full_report[Path(data_path).stem] = data_quality_verif_report

    with open(DATA_QUALITY_VERIFICATION_REPORT_PATH, "w") as f:
        json.dump(data_quality_verif_full_report, f, indent=4)

    # Data exploration
    data_expl_full_report = {}
    for data_path in data_paths:
        data_expl_report = explore_data(str(data_path))
        data_expl_full_report[Path(data_path).stem] = data_expl_report

    with open(DATA_EXPLORATION_REPORT_PATH, "w") as f:
        json.dump(data_expl_full_report, f, indent=4)

    ##################################################
    # [1] DATA PREPARATION
    ##################################################
    data_prep_result = prepare_data(data_paths)
    data_prep_report = data_prep_result["report"]

    with open(DATA_PREPARATION_REPORT_PATH, "w") as f:
        json.dump(data_prep_report, f, indent=4)

    ##################################################
    # [2] MODEL TRAINING
    ##################################################
