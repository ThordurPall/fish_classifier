# -*- coding: utf-8 -*-
# import json

# import joblib
# import numpy as np
from azureml.core.model import Model


# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = Model.get_model_path("fish-classifier-test")

    # Set the model to model_path for now
    model = model_path
    #  model = joblib.load(model_path)


# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    # sdata = np.array(json.loads(raw_data)["data"])

    # Get a prediction from the model
    # predictions = model.predict(data)

    # Get the corresponding classname for each prediction (0 or 1)
    # classnames = ["not-diabetic", "diabetic"]
    # predicted_classes = []
    # for prediction in predictions:
    #    predicted_classes.append(classnames[prediction])

    # Return the predictions as JSON
    # Just to check that we have access to the model here
    return model  # json.dumps(predicted_classes)
