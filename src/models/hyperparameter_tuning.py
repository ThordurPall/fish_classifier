# -*- coding: utf-8 -*-
import logging
import sys
from pathlib import Path

import hydra
import optuna
import plotly.io as pio
from omegaconf import OmegaConf

from src.models.train_model import train_model

project_dir = Path(__file__).resolve().parents[2]


@hydra.main(config_path=str(project_dir) + "/config", config_name="config")
def hyperparameter_tuning_hydra(config):
    log = logging.getLogger(__name__)
    log.info(f"Current configuration: \n {OmegaConf.to_yaml(config)}")
    bounds = config.optuna  # Current set of optuna bounds
    paths = config.paths
    log.info(f"Current Optuna configuration bounds: \n {bounds}")
    log.info(f"Current paths: \n {paths}")

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    # Set up the median stopping rule as the pruning condition
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3),
        direction="maximize",
    )
    study.optimize(
        lambda trial: optuna_objective(
            trial,
            paths=paths,
            optuna_settings=bounds,
        ),
        n_trials=3,
    )

    # Plot optimization history of all trials in a study
    training_figures_filepath = paths.training_figures_filepath
    fig = optuna.visualization.plot_optimization_history(study)
    pio.write_image(fig, training_figures_filepath + "optuna_optimization_history.pdf")

    # Plot intermediate values of all trials in a study -
    # Visualize the learning curves of the trials
    fig = optuna.visualization.plot_intermediate_values(study)
    pio.write_image(fig, training_figures_filepath + "optuna_accuracy_curve.pdf")

    # Plot the high-dimensional parameter relationships in a study
    fig = optuna.visualization.plot_parallel_coordinate(study)
    pio.write_image(
        fig, training_figures_filepath + "optuna_high_dim_par_relationships.pdf"
    )

    # Plot the parameter relationship as contour plot in a study
    fig = optuna.visualization.plot_contour(study)
    pio.write_image(
        fig, training_figures_filepath + "optuna_contour_par_relationships.pdf"
    )

    # Visualize individual hyperparameters as slice plot
    fig = optuna.visualization.plot_slice(study)
    pio.write_image(fig, training_figures_filepath + "optuna_individual_pars.pdf")

    # Plot hyperparameter importances
    fig = optuna.visualization.plot_param_importances(study)
    pio.write_image(
        fig, training_figures_filepath + "optuna_individual_par_importance.pdf"
    )

    # Plot the objective value EDF (empirical distribution function) of a study
    fig = optuna.visualization.plot_edf(study)
    pio.write_image(fig, training_figures_filepath + "optuna_edf.pdf")


def optuna_objective(
    trial,
    paths,
    optuna_settings,
):
    # Suggest a set of hyperparameters
    learning_rate = trial.suggest_loguniform(
        "learning_rate",
        optuna_settings.learning_rate.min,
        optuna_settings.learning_rate.max,
    )
    dropout_p = trial.suggest_uniform(
        "dropout_p", optuna_settings.dropout_p.min, optuna_settings.dropout_p.max
    )
    batch_size = trial.suggest_discrete_uniform(
        "batch_size",
        optuna_settings.batch_size.min,
        optuna_settings.batch_size.max,
        optuna_settings.batch_size.discretization_step,
    )

    print(f"Current learning rate: \n {learning_rate}")
    print(f"Current dropout: \n {dropout_p}")
    print(f"Current batch size: \n {batch_size}")

    train_val_dict = train_model(
        trained_model_filepath=paths.trained_model_filepath,
        training_statistics_filepath=paths.training_statistics_filepath,
        training_figures_filepath=paths.training_figures_filepath,
        epochs=optuna_settings.epochs,
        learning_rate=learning_rate,
        dropout_p=dropout_p,
        batch_size=batch_size,
        seed=optuna_settings.seed,
        trial=trial,
    )
    return train_val_dict["val_accuracies"][-1]


if __name__ == "__main__":
    hyperparameter_tuning_hydra()
