"""src/main.py

Main execution script.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from config import (
    early_stopping_config,
    loss_alpha,
    model_config,
    optimizer_config,
    scheduler_config,
    seed,
    training_batch_size,
    training_device,
    training_epochs,
)
from const import (
    BEST_MODEL_PATH,
    DATA_EXPLORATION_REPORT_PATH,
    DATA_FORMAT,
    DATA_PATH,
    DATA_PREPARATION_REPORT_PATH,
    DATA_QUALITY_VERIFICATION_REPORT_PATH,
    MODELING_TRAINING_REPORT_PATH,
)
from data.explore import explore_data
from data.prepare import prepare_data
from data.verify_quality import verify_data_quality
from modeling.train import Trainer


def main() -> None:
    """Main execution function.

    Returns:
        None
    """
    data_paths = list(DATA_PATH.glob(DATA_FORMAT))

    ##################################################
    # [PRE] REPRODUCIBILITY
    ##################################################
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

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
    trainer = Trainer(
        model_config=model_config,
        model_path=Path(BEST_MODEL_PATH),
        device=training_device,
    )
    training_report = trainer.train(
        X_train=data_prep_result["X_train"],
        y_train=data_prep_result["y_train"],
        A_train=data_prep_result["A_train"],
        X_val=data_prep_result["X_valid"],
        y_val=data_prep_result["y_valid"],
        A_val=data_prep_result["A_valid"],
        batch_size=training_batch_size,
        training_epochs=training_epochs,
        loss_alpha=loss_alpha,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        early_stopping_config=early_stopping_config,
        prep_report=data_prep_report,
    )

    with open(MODELING_TRAINING_REPORT_PATH, "w") as f:
        json.dump(training_report, f, indent=4)


if __name__ == "__main__":
    main()
