# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchdrift
from sklearn.manifold import TSNE, Isomap
from torch.utils.data import random_split
from torchdrift.detectors.mmd import (ExpKernel, GaussianKernel,
                                      RationalQuadraticKernel)

from src.models.Classifier import Classifier
from src.models.Hyperparameters import Hyperparameters as hp


def plot_tsne_test_set(trained_model_filepath, data_file_path, figures_folderpath):
    """ Extracts features just before the final classification layer of the network
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
        dropout_p=0.25,
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
    """ Plots the training set and test set class distributions """
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


def drift_detection(
    training_data_filepath,
    test_data_filepath,
    trained_model_filepath,
    figures_folderpath,
    plot_examples=False,
    detector_num_batches=100,
):
    """ Implements drift detection with the fish dataset """
    logger = logging.getLogger(__name__)
    logger.info("Drift detection with the fish dataset")

    # Check if there is a GPU available to use
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the training data
    project_dir = Path(__file__).resolve().parents[2]
    train_set_path = str(project_dir) + training_data_filepath
    train_imgs, train_labels = torch.load(train_set_path)
    train_set = torch.utils.data.TensorDataset(train_imgs, train_labels)

    # Split the training data in training and validation set
    train_n = int(0.75 * len(train_set))
    val_n = len(train_set) - train_n
    train_data, val_data = random_split(train_set, [train_n, val_n])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    batch_size = int(np.min([len(val_data), 512]))
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=True
    )
    print(f"Length of Train Data : {len(train_data)}")
    print(f"Length of Validation Data : {len(val_data)}")

    # Load the test data
    test_set_path = str(project_dir) + test_data_filepath
    test_imgs, test_labels = torch.load(test_set_path)
    test_set = torch.utils.data.TensorDataset(test_imgs, test_labels)
    batch_size = int(np.min([len(test_set), 512]))
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    logger.info(f"Length of Test Data : {len(test_set)}")

    # Load images and labels
    images, labels = iter(val_loader).next()
    images_test, _ = iter(test_loader).next()
    images_corrupt = corruption_function(images)

    if plot_examples:
        # Plot example images
        N = 6
        plt.figure(figsize=(15, 5))
        for i in range(N):
            # Plot the fish images
            plt.subplot(2, N, i + 1)
            plt.title(labels[i].item())
            plt.imshow(images[i].permute(1, 2, 0))
            plt.xticks([])
            plt.yticks([])

            # Plot the fish images with Gaussian blur
            plt.subplot(2, N, i + 1 + N)
            plt.title(labels[i].item())
            plt.imshow(images_corrupt[i].permute(1, 2, 0))
            plt.xticks([])
            plt.yticks([])
        plt.show()

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
        dropout_p=0.25,
    )
    model.only_return_features = True
    state_dict = torch.load(
        project_dir.joinpath(trained_model_filepath), map_location=torch.device(device)
    )
    model.load_state_dict(state_dict)
    model.eval()  # Sets the model to evaluation mode

    # The drift detector - Using the Kernel MMD drift detector on
    # the features extracted by the pretrained model
    gaussian_kernel = torchdrift.detectors.KernelMMDDriftDetector(
        kernel=GaussianKernel()
    )
    exp_kernel = torchdrift.detectors.KernelMMDDriftDetector(kernel=ExpKernel())
    rational_quadratic_kernel = torchdrift.detectors.KernelMMDDriftDetector(
        kernel=RationalQuadraticKernel()
    )

    kernel_names = ["GaussianKernel", "ExpKernel", "RationalQuadraticKernel"]
    (scores_real, p_vals_real, scores_corrupt, p_vals_corrupt, scores_test, p_test,) = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for i, kernel in enumerate(
        [gaussian_kernel, exp_kernel, rational_quadratic_kernel]
    ):
        kernel_name = kernel_names[i]
        drift_detector = kernel
        print(kernel_name)

        # Fit the drift detector using training data
        torchdrift.utils.fit(
            train_loader, model, drift_detector, num_batches=detector_num_batches
        )

        # Test the output on actual validation inputs
        score, p_val = calculate_plot_drift(
            model(images),
            drift_detector,
            f"{kernel_name} real validation data",
            project_dir.joinpath(figures_folderpath).joinpath(
                kernel_name + "_Distributions_Real_Validation_Data.pdf"
            ),
        )
        scores_real.append(score)
        p_vals_real.append(p_val)

        # Test the output on corrupt validation inputs
        score, p_val = calculate_plot_drift(
            model(images_corrupt),
            drift_detector,
            f"{kernel_name} corrupt validation data",
            project_dir.joinpath(figures_folderpath).joinpath(
                kernel_name + "_Distributions_Corrupt_Validation_Data.pdf"
            ),
        )
        scores_corrupt.append(score)
        p_vals_corrupt.append(p_val)

        # Test the output on actual test inputs
        score, p_val = calculate_plot_drift(
            model(images_test),
            drift_detector,
            f"{kernel_name} real test data",
            project_dir.joinpath(figures_folderpath).joinpath(
                kernel_name + "_Distributions_Real_Test_Data.pdf"
            ),
        )
        scores_test.append(score)
        p_test.append(p_val)

    return {
        "ScoreValidationReal": scores_real,
        "ScoreValidationCorrupt": scores_corrupt,
        "ScoreTestReal": scores_test,
        "PValuesValidationReal": p_vals_real,
        "PValuesValidationCorrupt": p_vals_corrupt,
        "PValuesTestReal": p_test,
    }


def corruption_function(x: torch.Tensor):
    """ Applies the Gsaussian blur to x """
    return torchdrift.data.functional.gaussian_blur(x, severity=5)


def calculate_plot_drift(features, drift_detector, figure_title, figure_file_path):
    """ Calculates test statistic and p-value and visualizes the distributions """
    score = drift_detector(features)
    p_val = drift_detector.compute_p_value(features)

    # Visualize the two distribution to detemine if the look close
    mapper = Isomap(n_components=2)
    base_embedded = mapper.fit_transform(drift_detector.base_outputs)
    features_embedded = mapper.transform(features.detach().numpy())
    f = plt.figure(figsize=(12, 8))
    plt.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c="r")
    plt.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
    plt.title(f"{figure_title}, score {score:.2f} p-value {p_val:.2f}")
    f.savefig(
        figure_file_path, bbox_inches="tight",
    )
    return score, p_val
