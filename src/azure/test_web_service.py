# -*- coding: utf-8 -*-
import json

import requests
from azureml.core import Webservice, Workspace


def main():

    # Load the workspace from the saved config file
    ws = Workspace.from_config()
    print("Ready to use Azure ML to work with {}".format(ws.name))

    for webservice in ws.webservices:
        print(webservice)

    x_new = [
        [2, 180, 74, 24, 21, 23.9091702, 1.488172308, 22],
        [0, 148, 58, 11, 179, 39.19207553, 0.160829008, 45],
    ]

    # Convert the array to a serializable list in a JSON document
    input_json = json.dumps({"data": 1})

    # Set the content type
    headers = {"Content-Type": "application/json"}

    # Get the fish classifier web service
    webservice = Webservice(ws, "fish-classifier-service")
    endpoint = webservice.scoring_uri
    print(endpoint)

    predictions = requests.post(endpoint, input_json, headers=headers)
    print(predictions)

    # webservice.delete()
    # print("Service deleted.")


if __name__ == "__main__":
    main()
