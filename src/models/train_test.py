# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from src.models.evaluate_model import evaluate_model
from src.models.train_model import train_model

project_dir = Path(__file__).resolve().parents[2]
project_dir_str = str(project_dir)


@hydra.main(config_path="config", config_name="config")
def train_test(config):
    log = logging.getLogger(__name__)
    log.info(f"Current configuration: \n {OmegaConf.to_yaml(config)}")
    bounds = config.final_model  # Get the final model parameters
    paths = config.paths
    log.info(f"Current Optuna configuration bounds: \n {bounds}")
    log.info(f"Current paths: \n {paths}")

    # Train the final model
    _ = train_model(
        trained_model_filepath=paths.trained_model_filepath,
        training_statistics_filepath=paths.training_statistics_filepath,
        training_figures_filepath=paths.training_figures_filepath,
        use_azure=bounds.use_azure,
        epochs=bounds.epochs,
        learning_rate=bounds.learning_rate,
        dropout_p=bounds.dropout_p,
        batch_size=bounds.batch_size,
        seed=bounds.seed,
    )

    # Evaluate the final model
    model_path = project_dir.joinpath(paths.trained_model_filepath)
    if bounds.use_azure:
        model_path = "./outputs/" + paths.trained_model_filepath
    test_accuracy = evaluate_model(
        trained_model_filepath=model_path, batch_size=bounds.batch_size
    )
    log.info(f"Final test accuracy: {test_accuracy}")


if __name__ == "__main__":
    train_test()
