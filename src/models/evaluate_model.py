# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path

import torch

from src.models.Classifier import Classifier
from src.models.Hyperparameters import Hyperparameters as hp


def evaluate_model(
    trained_model_filepath="models/trained_model.pth", batch_size=64, num_workers=0
):
    """Evaluates the trained network using test subset of the fish dataset"""
    logger = logging.getLogger(__name__)
    logger.info("Evaluating a trained network using a test subset")

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
        hype["activation"],
    )
    project_dir = Path(__file__).resolve().parents[2]
    state_dict = torch.load(
        project_dir.joinpath(trained_model_filepath), map_location=torch.device(device)
    )
    model.load_state_dict(state_dict)

    # Define a transform to normalize the data
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                transforms.Normalize((0.5,), (0.5,)), ])

    # Get the class id, label mapping
    mapping = {}
    mapping_file_path = str(project_dir) + "/data/processed/mapping.json"
    with open(mapping_file_path) as json_file:
        mapping = json.load(json_file)

    # Load the test data
    test_set_path = str(project_dir) + "/data/processed/test.pt"
    test_imgs, test_labels = torch.load(test_set_path)
    test_set = torch.utils.data.TensorDataset(test_imgs, test_labels)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    logger.info(f"Length of Test Data : {len(test_set)}")

    # Evaluate test performance
    test_correct = 0

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        model.eval()  # Sets the model to evaluation mode

        # Run through all the test points
        for images, labels in test_loader:
            # Transfering images and labels to GPU if available
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            log_ps = model(images)
            ps = torch.exp(log_ps)

            # Keep track of how many are correctly classified
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_correct += equals.type(torch.FloatTensor).sum().item()
        test_accuracy = test_correct / len(test_set)
    return test_accuracy
