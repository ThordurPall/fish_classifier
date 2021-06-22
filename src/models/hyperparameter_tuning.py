# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import hydra
import optuna
from omegaconf import OmegaConf

project_dir = Path(__file__).resolve().parents[2]


@hydra.main(config_path=str(project_dir) + "/config", config_name="config")
def hyperparameter_tuning_hydra(config):
    log = logging.getLogger(__name__)
    log.info(f"Current configuration: \n {OmegaConf.to_yaml(config)}")
    print(config)
    bounds = config.optuna  # Current set of optuna bounds
    print(bounds)
    learning_rate_min = bounds.learning_rate.min
    print(learning_rate_min)
    print(bounds.learning_rate.max)


if __name__ == "__main__":
    hyperparameter_tuning_hydra()

