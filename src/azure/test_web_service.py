# -*- coding: utf-8 -*-
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt

import requests
import torch
from azureml.core import Webservice, Workspace
from PIL import Image
from torchvision import transforms

from src.models.Hyperparameters import Hyperparameters as hp


def main():

    # Load the workspace from the saved config file
    ws = Workspace.from_config()
    print("Ready to use Azure ML to work with {}".format(ws.name))

    # Show available web services
    for webservice in ws.webservices:
        print(webservice)

    # Get the fish classifier web service and its uri
    # webservice = Webservice(ws, "fish-classifier-service")
    # endpoint = webservice.scoring_uri
    # print(endpoint)

    # Test the web service with a single test image
    # Open the image and change color encoding to RGB
    # in case the image is a png
    img_name = "./data/raw/unzipped/NA_Fish_Dataset/Black Sea Sprat/00001.png"
    img = Image.open(img_name)
    print(img)
    if img_name.endswith(".png"):
        img = img.convert("RGB")
    print(img)
    print(img.size)
    img_tensor = transforms.ToTensor()(img)  # .unsqueeze_(0)
    print(img_tensor.shape)
    print(img_tensor)
    img_numpy = img_tensor.numpy()
    print(img_numpy.shape)
    print(img_numpy.ravel().shape)
    img_list = img_numpy.tolist()
    # print(img_list)

    # hype = hp().config
    # img_tensor = torch.Tensor(1, 3, hype["image_height"], hype["image_width"])

    # Convert the array to a serializable list in a JSON document
    # input_json = json.dumps({"data": img_tensor})

    data = {}
    with open(img_name, mode="rb") as file:
        img = file.read()
    data["img"] = base64.b64encode(img).decode("utf-8")
    # Set the content type
    headers = {"Content-Type": "application/json"}
    input_json = json.dumps(data)

    # predictions = requests.post(endpoint, input_json, headers=headers)
    # print(print(predictions.text))
    # print(input_json)

    json_payload = json.loads(input_json)
    img_byte = json_payload["img"]
    # json_img = json.loads(json.dumps(jsonStr))
    img_64 = base64.b64decode(img_byte)

    img = BytesIO(img_64)
    image = Image.open(img)
    plt.imshow(image)
    plt.show()
    # print(input_json["img"])

    # webservice.delete()
    # print("Service deleted.")

    # Test the web service on the entire test set


if __name__ == "__main__":
    main()
