# -*- coding: utf-8 -*-
import json

import numpy as np
import torch
from azureml.core.model import Model

from src.models.Classifier import Classifier
from src.models.Hyperparameters import Hyperparameters as hp


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
    # Transform the input data to a numpy array and then to tensor
    img = raw_data["img"]
    # data_np = np.array(json.loads(raw_data)["data"])
    # data = torch.from_numpy(data_np)
    print(img)

    # Get a prediction from the model
    # predictions = model.predict(data)

    # Get the corresponding classname for each prediction (0 or 1)
    # classnames = ["not-diabetic", "diabetic"]
    # predicted_classes = []
    # for prediction in predictions:
    #    predicted_classes.append(classnames[prediction])

    # Return the predictions as JSON
    # Just to check that we have access to the model here
    return img  # json.dumps(predicted_classes)
