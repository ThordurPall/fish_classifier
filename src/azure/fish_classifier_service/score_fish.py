# -*- coding: utf-8 -*-
import json

import torch
from azureml.core.model import Model

from src.models.Classifier import Classifier
from src.models.Hyperparameters import Hyperparameters as hp
from src.utils.AugmentationPipeline import AugmentationPipeline
from src.utils.DataTransforms import DataTransforms


# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file
    model_path = Model.get_model_path("fish-classifier-test")

    # Check if there is a GPU available to use
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    hype = hp().config
    model = Classifier(
        hype["num_classes"],
        hype["filter1_in"],
        hype["filter1_out"],
        hype["filter2_out"],
        hype["filter3_out"],
        hype["image_height"],
        hype["image_width"],
        hype["pad"],
        hype["stride"],
        hype["kernel"],
        hype["pool"],
        hype["fc_1"],
        hype["fc_2"],
    )
    state_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()  # Sets the model to evaluation mode


# Called when a request is received
def run(raw_data):
    # Read in image form json, decode and transform to tensor
    dt = DataTransforms()
    image = dt.b64_to_PIL_image(json.loads(raw_data)["img"])
    image = dt.PIL_image_to_tensor(image)
    image = image.unsqueeze(0)

    # Use the model to get predictions
    log_ps = model(image)
    ps = torch.exp(log_ps)

    # Get the most probable class and its probability
    top_probs, top_class = ps.topk(1, dim=1)

    # Define a mapping from class IDs to labels
    classes = {
        "0": "Trout",
        "1": "Shrimp",
        "2": "Striped Red Mullet",
        "3": "Gilt Head Bream",
        "4": "Black Sea Sprat",
        "5": "Sea Bass",
        "6": "Red Sea Bream",
        "7": "Red Mullet",
        "8": "Horse Mackerel",
    }
    return json.dumps(
        {"Class": classes[str(top_class.item())], "Probability": str(top_probs.item())}
    )
