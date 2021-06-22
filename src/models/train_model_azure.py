# -*- coding: utf-8 -*-
import os

import click
from azureml.core import (ComputeTarget, Environment, Experiment, Model,
                          ScriptRunConfig, Workspace)
from azureml.core.conda_dependencies import CondaDependencies


@click.option(
    "-uo/-no-uo",
    "--use_optuna/--no_use_optuna",
    type=bool,
    default=False,
    help="Set to True to use Optuna for hyperparameter tuning (default is False)",
)
def main(use_optuna):
    # Create a Python environment for the experiment
    env = Environment("experiment-fish-classifier-test")

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
        ],
    )
    whl_path = "./dist/src-0.1.15-py3-none-any.whl"
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
    else:
        script = "train_model_command_line.py"
        e = 30
        lr = 0.001
        script_args = ["--epochs", e, "--learning_rate", lr, "--use_azure", True]
    script_config = ScriptRunConfig(
        source_directory=experiment_folder,
        script=script,
        environment=env,
        arguments=script_args,
        compute_target=compute_target,
    )

    # Create and submit the experiment
    experiment = Experiment(workspace=ws, name="fish-classifier-training-test")
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
            model_name="fish-classifier-test",
            tags={"Training data": "fish-classifier-test"},
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
