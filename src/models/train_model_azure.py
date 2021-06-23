# -*- coding: utf-8 -*-
import glob
import os.path

import click
from azureml.core import (
    ComputeTarget,
    Environment,
    Experiment,
    Model,
    ScriptRunConfig,
    Workspace,
)
from azureml.core.conda_dependencies import CondaDependencies


@click.command()
@click.option(
    "-uo/-no-uo",
    "--use_optuna/--no_use_optuna",
    type=bool,
    default=False,
    help="Set to True to use Optuna for hyperparameter tuning (default is False)",
)
def main(use_optuna):

    # Create a Python environment for the experiment
    env = Environment("Train-initial-model")

    # Load the workspace from the saved config file
    ws = Workspace.from_config()
    print("Ready to use Azure ML to work with {}".format(ws.name))

    # Set the compute target
    compute_target = ComputeTarget(ws, "FirstMachine1")
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

    if use_optuna:
        script_args = None
        script = "hyperparameter_tuning.py"
    else:
        script = "train_model_command_line.py"
        e = 30
        lr = 0.001
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
    experiment = Experiment(workspace=ws, name="Train-initial-model")
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
            "epochs": e,
            "learning_rate": lr,
            "Final train loss": metrics["Train loss"][-1],
            "Final train accuracy": metrics["Train accuracy"][-1],
            "Final validation loss": metrics["Validation loss"][-1],
            "Final validation accuracy": metrics["Validation accuracy"][-1],
        }
        run.register_model(
            model_path="./outputs/models/trained_model.pth",
            model_name="Train-initial-model",
            tags={"Training data": "Train-initial-model"},
            properties=model_props,
        )

        # List registered models
        for model in Model.list(ws):
            print(model.name, "version:", model.version)
            for tag_name in model.tags:
                tag = model.tags[tag_name]
                print("\t", tag_name, ":", tag)
            for prop_name in model.properties:
                prop = model.properties[prop_name]
                print("\t", prop_name, ":", prop)
            print("\n")

    # Download files in the "outputs" folder and store locally
    download_folder = "azure-downloaded-files"
    run.download_files(prefix="outputs", output_directory=download_folder)

    # Verify the files have been downloaded
    for root, directories, filenames in os.walk(download_folder):
        for filename in filenames:
            print(os.path.join(root, filename))


if __name__ == "__main__":
    main()
