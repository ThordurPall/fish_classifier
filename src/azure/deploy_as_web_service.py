# -*- coding: utf-8 -*-
import os

from azureml.core import Environment, Model, Workspace
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice


def main():
    # Load the workspace from the saved config file
    ws = Workspace.from_config()
    print("Ready to use Azure ML to work with {}".format(ws.name))

    # Get the latest fish classifier model
    model = Model(ws, name="fish-classifier")
    print(model)

    # Set path for scoring script
    experiment_folder = "./src/azure/fish_classifier_service/"
    script_file = os.path.join(experiment_folder, "score_fish.py")

    # Ensure the required packages are installed
    packages = CondaDependencies.create(
        conda_packages=["pip"],
        pip_packages=[
            "azureml-defaults",
            "torch",
            "torchvision",
            "pillow",
            "kornia",
            "numpy",
        ],
    )
    whl_path = "./dist/src-0.1.14-py3-none-any.whl"
    whl_url = Environment.add_private_pip_wheel(
        workspace=ws, exist_ok=True, file_path=whl_path
    )
    packages.add_pip_package(whl_url)

    # Save the environment config as a .yml file
    env_file = os.path.join(experiment_folder, "fish_classifier_env.yml")
    with open(env_file, "w") as f:
        f.write(packages.serialize_to_string())
    print("Saved dependency info in", env_file)

    # Configure the scoring environment
    inference_config = InferenceConfig(
        runtime="python", entry_script=script_file, conda_file=env_file
    )

    # Finally, deploy it as a web service
    deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
    service_name = "fish-classifier-service"
    service = Model.deploy(
        ws, service_name, [model], inference_config, deployment_config
    )

    service.wait_for_deployment(True)
    print(service.state)


if __name__ == "__main__":
    main()
