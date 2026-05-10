"""src/main.py

Main execution script.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from src.config import SEED, TRAINING_DEVICE
from src.const import (
    DATA_EXPLORATION_REPORT_PATH,
    DATA_FORMAT,
    DATA_PATH,
    DATA_PREPARATION_REPORT_PATH,
    DATA_QUALITY_VERIFICATION_REPORT_PATH,
    MODELING_TRAINING_REPORT_PATH,
)
from src.data.explore import explore_data
from src.data.prepare import prepare_data
from src.data.verify_quality import verify_data_quality
from src.modeling.train import TimeSeriesTrainer


def main() -> None:
    """Main execution function.

    This function orchestrates the entire workflow, including:
    1. Data quality verification and exploration
    2. Data preparation
    3. Model training
    4. Model evaluation

    Returns:
        None
    """
    data_paths = list(DATA_PATH.glob(DATA_FORMAT))

    ##################################################
    # [PRE] REPRODUCIBILITY
    ##################################################
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(SEED)
    generator = torch.Generator()
    generator.manual_seed(SEED)

    ##################################################
    # [1] DATA QUALITY VERIFICATION AND EXPLORATION
    ##################################################
    # Data quality verification
    data_quality_verif_full_report = {}
    for data_path in data_paths:
        data_quality_verif_report = verify_data_quality(str(data_path))
        data_quality_verif_full_report[Path(data_path).stem] = (
            data_quality_verif_report
        )

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
    # [2] DATA PREPARATION
    ##################################################
    data_prep_result = prepare_data(data_paths)
    data_prep_report = data_prep_result["report"]

    with open(DATA_PREPARATION_REPORT_PATH, "w") as f:
        json.dump(data_prep_report, f, indent=4)

    ##################################################
    # [3] MODEL TRAINING
    ##################################################
    trainer = TimeSeriesTrainer(device=TRAINING_DEVICE)
    training_report = trainer.train(
        X_train=data_prep_result["X_train"],
        y_train=data_prep_result["y_train"],
        A_train=data_prep_result["A_train"],
        batch_size=4,
    )

    with open(MODELING_TRAINING_REPORT_PATH, "w") as f:
        json.dump(training_report, f, indent=4)
