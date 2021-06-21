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
    hype = config.optuna  # Current set of hyperparameters
    print(hype)


if __name__ == "__main__":
    hyperparameter_tuning_hydra()

