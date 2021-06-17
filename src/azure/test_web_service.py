# -*- coding: utf-8 -*-

import json

import matplotlib.pyplot as plt
import requests
from azureml.core import Webservice, Workspace
from PIL import Image

from src.utils.DataTransforms import DataTransforms


def main():

    # Load the workspace from the saved config file
    dt = DataTransforms()
    ws = Workspace.from_config()
    print("Ready to use Azure ML to work with {}".format(ws.name))

    # Show available web services
    for webservice in ws.webservices:
        print(webservice)

    # Get the fish classifier web service and its uri
    webservice = Webservice(ws, "fish-classifier-service")
    endpoint = webservice.scoring_uri
    print(endpoint)

    # Test the web service with a single test image
    # Open the image and change color encoding to RGB
    # in case the image is a png
    img_name = "./data/raw/unzipped/NA_Fish_Dataset/Black Sea Sprat/00015.png"
    img = Image.open(img_name)
    img = img.convert("RGB")

    # Convert the image to base64
    data = {}
    data["img"] = dt.PIL_image_to_b64(Image.open(img_name))

    # Set the content type
    headers = {"Content-Type": "application/json"}
    input_json = json.dumps(data)

    # Call the REST service
    predictions = requests.post(endpoint, input_json, headers=headers)
    print(predictions)
    print(predictions.text)

    # webservice.delete()
    # print("Service deleted.")

    # Test the web service on the entire test set


if __name__ == "__main__":
    main()
