# -*- coding: utf-8 -*-
import glob
import os.path

import click
from azureml.core import (ComputeTarget, Environment, Experiment,
                          ScriptRunConfig, Workspace)
from azureml.core.conda_dependencies import CondaDependencies


@click.command()
@click.option(
    "-uo/-no-uo",
    "--use_optuna/--no_use_optuna",
    type=bool,
    default=False,
    help="Set to True to use Optuna for hyperparameter tuning (default is False)",
)
@click.option(
    "-tf/-no-tf",
    "--train_final/--no_train_final",
    type=bool,
    default=False,
    help="Set to True to trian the final model (default is False)",
)
def main(use_optuna, train_final):
    # Create a Python environment for the experiment
    # env = Environment("experiment-fish-classifier-final-model")
    env = Environment("experiment-fish-classifier-hyperparameter-tuning")

    # Load the workspace from the saved config file
    ws = Workspace.from_config()
    print("Ready to use Azure ML to work with {}".format(ws.name))

    # Set the compute target
    compute_target = ComputeTarget(ws, "MLOpsGPU")
    print("Ready to use compute target: {}".format(compute_target.name))

    # Ensure the required packages are installed
    packages = CondaDependencies.create(
        conda_packages=["pip"],
        pip_packages=[
            "azureml-defaults",
            "torch",
            "torchvision",
            "pandas",
            "numpy",
            "matplotlib",
            "kornia",
            "gdown",
            "pillow",
            "optuna",
            "hydra-core",
            "sklearn",
        ],
    )

    folder_path = "./dist"
    file_type = "/*"
    files = glob.glob(folder_path + file_type)

    latest_whl = max(files, key=os.path.getctime)

    whl_path = latest_whl

    whl_url = Environment.add_private_pip_wheel(
        workspace=ws, exist_ok=True, file_path=whl_path
    )
    packages.add_pip_package(whl_url)
    env.python.conda_dependencies = packages

    # Create a script config for training
    experiment_folder = "./src/models"

    script_args = None
    if use_optuna:
        script = "hyperparameter_tuning.py"
    elif train_final:
        script = "train_test.py"
    else:
        script = "train_model_command_line.py"
        e = 50
        lr = 0.00038434
        dropout_p = 0.0
        script_args = [
            "--epochs",
            e,
            "--learning_rate",
            lr,
            "--use_azure",
            True,
            "--dropout_p",
            dropout_p,
        ]

    script_config = ScriptRunConfig(
        source_directory=experiment_folder,
        script=script,
        environment=env,
        arguments=script_args,
        compute_target=compute_target,
    )

    # Create and submit the experiment
    # experiment = Experiment(workspace=ws, name="experiment-fish-classifier-final-model")
    experiment = Experiment(
        workspace=ws, name="experiment-fish-classifier-hyperparameter-tuning"
    )
    run = experiment.submit(config=script_config)

    # Block until the experiment run has completed
    run.wait_for_completion()
    print("Finished running the training script")

    if not use_optuna:
        # Get logged metrics and files
        print("Getting run metrics")
        metrics = run.get_metrics()
        for key in metrics.keys():
            print(key, metrics.get(key))

        print("\n")

        print("Getting run files")
        for file in run.get_file_names():
            print(file)

        # Register the model
        model_props = {
            "Final train loss": metrics["Train loss"][-1],
            "Final train accuracy": metrics["Train accuracy"][-1],
            "Final validation loss": metrics["Validation loss"][-1],
            "Final validation accuracy": metrics["Validation accuracy"][-1],
        }

        # Verify the files have been downloaded
        model_path = "./outputs/models/trained_model.pth"
        if train_final:
            # Get the correct path to the model
            model_path = add_train_final_dir()
        run.register_model(
            model_path=model_path,
            model_name="fish-classifier",
            tags={"Training data": "fish-classifier"},
            properties=model_props,
        )
    # Download files in the "outputs" folder and store locally
    download_folder = "azure-downloaded-files"
    run.download_files(prefix="outputs", output_directory=download_folder)

    # Verify the files have been downloaded
    for root, directories, filenames in os.walk(download_folder):
        for filename in filenames:
            print(os.path.join(root, filename))


def add_train_final_dir():
    # TODO
    model_path = "./outputs/"
    for root, directories, filenames in os.walk(model_path, topdown=True):
        model_path += directories[-1]
        break
    model_path += "/"
    for root, directories, filenames in os.walk(model_path, topdown=True):
        model_path += directories[-1]
        break
    model_path += "/outputs/models/trained_model.pth"
    return model_path


if __name__ == "__main__":
    main()
