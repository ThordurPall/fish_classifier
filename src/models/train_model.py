import json
import logging
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from azureml.core import Run
from torch import nn, optim
from torch.utils.data import random_split

from src.data.MakeDataset import MakeDataset
from src.models.Classifier import Classifier
from src.models.Hyperparameters import Hyperparameters as hp


def train_model(
    trained_model_filepath,
    training_statistics_filepath,
    training_figures_filepath,
    use_azure=False,
    epochs=10,
    learning_rate=0.001,
):

    # Check if there is a GPU available to use
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if use_azure:
        make_data = MakeDataset(generated_images_per_image=1, image_size=12)
        make_data.make_dataset()
        print("Dataset created")

        # Get the experiment run context. That is, retrieve the experiment
        # run context when the script is run
        run = Run.get_context()
        run.log("Learning rate", learning_rate)
        run.log("Epochs", epochs)

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info("Training a fish classifier")

    project_dir = Path(__file__).resolve().parents[2]
    train_set_path = str(project_dir) + "/data/processed/training.pt"
    mapping_file_path = str(project_dir) + "/data/processed/mapping.json"
    train_imgs, train_labels = torch.load(train_set_path)  # img, label
    mapping = {}

    with open(mapping_file_path) as json_file:
        mapping = json.load(json_file)

    # load data
    train_set = torch.utils.data.TensorDataset(train_imgs, train_labels)

    # split data in training and validation set
    train_n = int(0.65 * len(train_set))
    val_n = len(train_set) - train_n
    train_data, val_data = random_split(train_set, [train_n, val_n])
    print(f"Length of Train Data : {len(train_data)}")
    print(f"Length of Validation Data : {len(val_data)}")

    ##### Hyper parameters
    batch_size = 64
    num_classes = len(mapping)
    rgb = train_imgs.shape[1]
    height = train_imgs.shape[2]
    width = train_imgs.shape[3]
    filter1_in = rgb
    filter1_out = 6
    kernel = 2
    pool = 2
    filter2_out = 16
    filter3_out = 48
    fc_1 = 120
    fc_2 = 84
    pad = 0
    stride = 1

    hype = hp().config

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=hype['batch_size'], shuffle=True, num_workers=0
    )  # changed num_workers to 0 because i was getting error

    valoader = torch.utils.data.DataLoader(
        val_data, batch_size=hype['batch_size'], shuffle=True, num_workers=0
    )


    # Initialize the model and transfer to GPU if available
    model = Classifier(hype['num_classes'], hype['filter1_in'],
                       hype['filter1_out'], hype['filter2_out'],
                       hype['filter3_out'], hype['image_height'],
                       hype['image_width'], hype['pad'],
                       hype['stride'], hype['kernel'], hype['pool'],
                       hype['fc_1'], hype['fc_2'])
    model = model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=hype['lr'])

    # Implement the training loop
    print("Start training")
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    for e in range(epochs):
        train_loss = 0
        train_correct = 0

        for images, labels in trainloader:
            # Transfering images and labels to GPU if available
            images, labels = images.to(device), labels.to(device)

            # Set model to training mode and zero
            #  gradients since they accumulated
            model.train()
            optimizer.zero_grad()

            # Make a forward pass through the network to get the logits
            log_ps = model(images)
            ps = torch.exp(log_ps)

            # Use the logits to calculate the loss
            loss = criterion(log_ps, labels.long())
            train_loss += loss.item()

            # Perform a backward pass through the network
            #  to calculate the gradients
            loss.backward()

            # Take a step with the optimizer to update the weights
            optimizer.step()

            # Keep track of how many are correctly classified
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            train_correct += equals.type(torch.FloatTensor).sum().item()
        else:
            # Compute validattion loss and accuracy
            val_loss = 0
            val_correct = 0

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                model.eval()  # Sets the model to evaluation mode
                for images, labels in valoader:
                    # Transfering images and labels to GPU if available
                    images, labels = images.to(device), labels.to(device)

                    # Forward pass and compute loss
                    log_ps = model(images)
                    ps = torch.exp(log_ps)
                    val_loss += criterion(log_ps, labels.long()).item()

                    # Keep track of how many are correctly classified
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    val_correct += equals.type(torch.FloatTensor).sum().item()

            # Store and print losses and accuracies
            train_losses.append(train_loss / len(trainloader))
            train_accuracies.append(train_correct / len(train_data))
            val_losses.append(val_loss / len(valoader))
            val_accuracies.append(val_correct / len(val_data))

            logger.info(
                str("Epoch: {}/{}.. ".format(e + 1, epochs))
                + str("Training Loss: {:.3f}.. ".format(train_losses[-1]))
                + str("Training Accuracy: {:.3f}.. ".format(train_accuracies[-1]))
                + str("Validation Loss: {:.3f}.. ".format(val_losses[-1]))
                + str("Validation Accuracy: {:.3f}.. ".format(val_accuracies[-1]))
            )

    # Set file paths depending on running locally or on Azure
    model_path = project_dir.joinpath(trained_model_filepath)
    dict_path = project_dir.joinpath(training_statistics_filepath).joinpath(
        "train_val_dict.pickle"
    )
    l_fig_path = project_dir.joinpath(training_figures_filepath).joinpath(
        "Training_Loss.pdf"
    )
    a_fig_path = project_dir.joinpath(training_figures_filepath).joinpath(
        "Training_Accuracy.pdf"
    )

    if use_azure:
        # Update model path and make sure it exists
        os.makedirs("./outputs", exist_ok=True)
        model_path = "./outputs/" + trained_model_filepath
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Update dictionary path
        dict_path = (
            "./outputs/" + training_statistics_filepath + "train_val_dict.pickle"
        )
        os.makedirs(os.path.dirname(dict_path), exist_ok=True)

        # Update figure paths
        figures_path = "./outputs/" + training_figures_filepath
        os.makedirs(figures_path, exist_ok=True)
        l_fig_path = figures_path + "Training_Loss.pdf"
        a_fig_path = figures_path + "Training_Accuracy.pdf"

        # Log the training and validation losses and accuracies
        run.log_list("Train loss", train_losses)
        run.log_list("Train accuracy", train_accuracies)
        run.log_list("Validation loss", val_losses)
        run.log_list("Validation accuracy", val_accuracies)

    # Save the trained network
    torch.save(model.state_dict(), model_path)

    # Save the training and validation losses and accuracies as a dictionary
    train_val_dict = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
    }

    with open(dict_path, "wb") as f:
        # Pickle the 'train_val_dict' dictionary using
        #  the highest protocol available
        pickle.dump(train_val_dict, f, pickle.HIGHEST_PROTOCOL)

    # Plot the training loss curve
    f = plt.figure(figsize=(12, 8))
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch number")
    plt.ylabel("Loss")
    plt.legend()
    if use_azure:
        run.log_image(name="Training loss curve", plot=f)
    f.savefig(l_fig_path, bbox_inches="tight")

    # Plot the training accuracy curve
    f = plt.figure(figsize=(12, 8))
    plt.plot(train_accuracies, label="Training accuracy")
    plt.plot(val_accuracies, label="Validation accuracy")
    plt.xlabel("Epoch number")
    plt.ylabel("Accuracy")
    plt.legend()
    if use_azure:
        run.log_image(name="Training accuracy curve", plot=f)
    f.savefig(a_fig_path, bbox_inches="tight")

    if use_azure:
        # Complete the run
        run.complete()
        print("Completed running the training expriment")

    return train_val_dict
