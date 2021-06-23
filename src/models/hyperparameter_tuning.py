# -*- coding: utf-8 -*-
import logging
import os
import sys
from pathlib import Path

import hydra
import optuna
from azureml.core import Run
from omegaconf import OmegaConf

from src.models.train_model import train_model

project_dir = Path(__file__).resolve().parents[2]
project_dir_str = str(project_dir)


@hydra.main(config_path="config", config_name="config")
def hyperparameter_tuning_hydra(config):
    log = logging.getLogger(__name__)
    log.info(f"Current configuration: \n {OmegaConf.to_yaml(config)}")
    bounds = config.optuna  # Current set of optuna bounds
    paths = config.paths
    training_figures_filepath = project_dir_str + "/" + paths.training_figures_filepath
    log.info(f"Current Optuna configuration bounds: \n {bounds}")
    log.info(f"Current paths: \n {paths}")

    if bounds.use_azure:
        # Get the experiment run context. That is, retrieve the experiment
        # run context when the script is run
        run = Run.get_context()

        # Create the right azure foldure structure
        os.makedirs("./outputs", exist_ok=True)
        training_figures_filepath = "./outputs/" + paths.training_figures_filepath
        os.makedirs(os.path.dirname(training_figures_filepath), exist_ok=True)

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    # Set up the median stopping rule as the pruning condition
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=bounds.n_startup_trials,
            n_warmup_steps=bounds.n_warmup_steps,
        ),
        direction="maximize",
    )
    study.optimize(
        lambda trial: optuna_objective(trial, paths=paths, optuna_settings=bounds,),
        n_trials=bounds.n_trials,
    )

    # Plot optimization history of all trials in a study
    print(training_figures_filepath)
    ax = optuna.visualization.matplotlib.plot_optimization_history(study)
    fig = ax.figure
    if bounds.use_azure:
        run.log_image(
            name="Optuna optimization history of all trials in a study", plot=fig
        )
    fig.savefig(
        training_figures_filepath + "optuna_optimization_history.pdf",
        bbox_inches="tight",
    )

    # Plot intermediate values of all trials in a study -
    # Visualize the learning curves of the trials
    ax = optuna.visualization.matplotlib.plot_intermediate_values(study)
    fig = ax.figure
    if bounds.use_azure:
        run.log_image(name="Optuna learning curves of the trials", plot=fig)
    fig.savefig(
        training_figures_filepath + "optuna_accuracy_curve.pdf", bbox_inches="tight",
    )

    # Plot hyperparameter importances
    ax = optuna.visualization.matplotlib.plot_param_importances(study)
    fig = ax.figure
    if bounds.use_azure:
        run.log_image(name="Optuna hyperparameter importances", plot=fig)
    fig.savefig(
        training_figures_filepath + "optuna_individual_par_importance.pdf",
        bbox_inches="tight",
    )

    if bounds.use_azure:
        # Complete the run
        run.complete()
        print("Completed running the training expriment")


def optuna_objective(
    trial, paths, optuna_settings,
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
    activation = trial.suggest_categorical("activation", optuna_settings.activation)

    print(f"Current learning rate: \n {learning_rate}")
    print(f"Current dropout: \n {dropout_p}")
    print(f"Current batch size: \n {batch_size}")

    train_val_dict = train_model(
        trained_model_filepath=paths.trained_model_filepath,
        training_statistics_filepath=paths.training_statistics_filepath,
        training_figures_filepath=paths.training_figures_filepath,
        use_azure=optuna_settings.use_azure,
        epochs=optuna_settings.epochs,
        learning_rate=learning_rate,
        dropout_p=dropout_p,
        batch_size=batch_size,
        seed=optuna_settings.seed,
        trial=trial,
        save_training_results=False,
        activation=activation,
    )
    return train_val_dict["val_accuracies"][-1]


if __name__ == "__main__":
    hyperparameter_tuning_hydra()
