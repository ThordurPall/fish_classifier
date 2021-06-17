# -*- coding: utf-8 -*-
import os

from azureml.core import Model, Workspace
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig


def main():
    # Load the workspace from the saved config file
    ws = Workspace.from_config()
    print("Ready to use Azure ML to work with {}".format(ws.name))

    # Get the latest fish classifier model
    model = Model(ws, name="fish-classifier-test")
    print(model)

    # Set path for scoring script
    experiment_folder = "./src/azure/fish_classifier_service/"
    script_file = os.path.join(experiment_folder, "score_fish.py")

    # Add the dependencies for the model (AzureML defaults is already included)
    myenv = CondaDependencies()
    myenv.add_conda_package("scikit-learn")

    # Save the environment config as a .yml file
    env_file = os.path.join(experiment_folder, "fish_classifier_env.yml")
    with open(env_file, "w") as f:
        f.write(myenv.serialize_to_string())
    print("Saved dependency info in", env_file)

    # Configure the scoring environment
    inference_config = InferenceConfig(
        runtime="python", entry_script=script_file, conda_file=env_file
    )

    deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
    service_name = "fish-classifier-service"
    service = Model.deploy(
        ws, service_name, [model], inference_config, deployment_config
    )

    service.wait_for_deployment(True)
    print(service.state)


if __name__ == "__main__":
    main()
