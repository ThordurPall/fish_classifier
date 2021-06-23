# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from src.models.Classifier import Classifier
from src.models.Hyperparameters import Hyperparameters as hp


def plot_tsne_test_set(trained_model_filepath, data_file_path, figures_folderpath):
    """Extracts features just before the final classification layer of the network
    in TRAINED_MODEL_FILEPATH and does t-SNE embedding of the features for
    the fish test set located in DATA_FILEPATH."""

    logger = logging.getLogger(__name__)
    logger.info("Creating a t-SNE embedding of the features for the fish test set")

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
        hype["dropout_p"],
    )
    model.also_return_features = True
    project_dir = Path(__file__).resolve().parents[2]
    state_dict = torch.load(
        project_dir.joinpath(trained_model_filepath), map_location=torch.device(device)
    )
    model.load_state_dict(state_dict)
    model.eval()  # Sets the model to evaluation mode

    # Get the class id, label mapping
    mapping = {}
    mapping_file_path = str(project_dir) + "/data/processed/mapping.json"
    with open(mapping_file_path) as json_file:
        mapping = json.load(json_file)

    # Load the test data
    test_set_path = str(project_dir) + data_file_path
    test_imgs, test_labels = torch.load(test_set_path)
    test_set = torch.utils.data.TensorDataset(test_imgs, test_labels)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    logger.info(f"Length of Test Data : {len(test_set)}")

    # Extracts features just before the final classification
    #  layer and do t-SNE embedding
    # Code from https://towardsdatascience.com/visualizing-feature-vectors
    # -embeddings-using-pca-and-t-sne-ef157cea3a42
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        test_imgs = torch.zeros(
            (0, 3, hype["image_height"], hype["image_width"]), dtype=torch.float32
        )
        test_predictions = []
        test_targets = []
        test_embeddings = torch.zeros((0, hype["fc_2"]), dtype=torch.float32)
        # x, y in test_loader:
        for images, labels in test_loader:
            # Transfering images and labels to GPU if available
            images, labels = images.to(device), labels.to(device)

            # Use the models to get the logits and the embeddings
            logits, embeddings = model(images)
            preds = torch.argmax(logits, dim=1)

            # Store the results
            test_predictions.extend(preds.tolist())
            test_targets.extend(labels.tolist())
            test_embeddings = torch.cat((test_embeddings, embeddings), 0)
            test_imgs = torch.cat((test_imgs, images), 0)
        test_imgs = np.array(test_imgs)
        test_embeddings = np.array(test_embeddings)
        test_targets = np.array(test_targets)
        test_predictions = np.array(test_predictions)
        print((test_predictions == test_targets).sum() / len(test_set))

    # Create a two dimensional t-SNE projection of the embeddings
    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(test_embeddings)

    # Plot the projected points as a scatter plot and label
    # them based on the pred labels
    cmap = plt.cm.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(12, 8))
    for lab in range(hype["num_classes"]):
        indices = test_predictions == lab
        ax.scatter(
            tsne_proj[indices, 0],
            tsne_proj[indices, 1],
            c=np.array(cmap(lab)).reshape(1, 4),
            label=mapping[str(lab)],
            alpha=0.8,
        )
    ax.legend(fontsize="large", markerscale=2)
    fig.savefig(
        project_dir.joinpath(figures_folderpath).joinpath("TSNE_test_set.pdf"),
        bbox_inches="tight",
    )


def plot_class_distributions(
    training_data_filepath, test_data_filepath, figures_folderpath
):
    """Plots the training set and test set class distributions"""
    # Load the training and test data
    project_dir = Path(__file__).resolve().parents[2]
    train_set_path = str(project_dir) + training_data_filepath
    _, train_labels = torch.load(train_set_path)
    test_set_path = str(project_dir) + test_data_filepath
    _, test_labels = torch.load(test_set_path)

    # Plot the data distribution of the fish train and test sets
    names = ["Training", "Test"]
    labels = [train_labels, test_labels]
    for i in range(2):
        f = plt.figure(figsize=(12, 8))
        plt.hist(labels[i].numpy(), density=False, bins=30)
        plt.ylabel("Count")
        plt.xlabel("Class ID")
        f.savefig(
            project_dir.joinpath(figures_folderpath).joinpath(
                names[i] + "_Class_Distribution.pdf"
            ),
            bbox_inches="tight",
        )
