import os
import pickle
from pathlib import Path

import optuna
import torch
from torch import nn, optim
from torch.utils.data import random_split

from src.data.MakeDataset import MakeDataset
from src.models.Classifier import Classifier
from src.models.Hyperparameters import Hyperparameters as hp


def objective(trial):
    LEARNING_RATE = trial.suggest_loguniform("LEARNING_RATE", low=1e-5, high=5e-2)
    BATCH_SIZE = trial.suggest_discrete_uniform("BATCH_SIZE", low=50, high=250, q=50)
    ACTIVATION = trial.suggest_categorical(
        "ACTIVATION_FUNCTION", ["relu", "leaky_relu"]
    )

    # DROPOUT = trial.suggest_float("DROPOUT", low=0.0, high=0.4, step=0.05)
    # OUTPUT_FEATURES = trial.suggest_discrete_uniform(
    #    "OUTPUT_FEATURES", low=16, high=512, q=100
    # )
    print(
        "LEARNING_RATE: ",
        LEARNING_RATE,
        "\tBATCH_SIZE: ",
        BATCH_SIZE,
        "\tACTIVATION: ",
        ACTIVATION,
        # "\tDROPOUT: ",
        #    DROPOUT,
        #    "\tOUTPUT_FEATURES: ",
        #    OUTPUT_FEATURES,
    )

    # Hyper parameters
    hype = hp().config

    # Check if there is a GPU available to use
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    project_dir = Path(__file__).resolve().parents[2]
    train_set_path = str(project_dir) + "/data/processed/training.pt"
    train_imgs, train_labels = torch.load(train_set_path)  # img, label

    # load data
    train_set = torch.utils.data.TensorDataset(train_imgs, train_labels)

    # split data in training and validation set
    train_n = int(0.85 * len(train_set))
    val_n = len(train_set) - train_n
    train_data, val_data = random_split(train_set, [train_n, val_n])

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=int(BATCH_SIZE), shuffle=True, num_workers=0
    )  # changed num_workers to 0 because i was getting error

    valoader = torch.utils.data.DataLoader(
        val_data, batch_size=int(BATCH_SIZE), shuffle=True, num_workers=0
    )

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Initialize the model and transfer to GPU if available
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
        ACTIVATION,
    )
    model = model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Implement the training loop
    print("Start training")
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    for e in range(hype["epochs"]):
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
            _, top_class = ps.topk(1, dim=1)
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
                    _, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    val_correct += equals.type(torch.FloatTensor).sum().item()

            # Store and print losses and accuracies
            train_losses.append(train_loss / len(trainloader))
            train_accuracies.append(train_correct / len(train_data))
            val_losses.append(val_loss / len(valoader))
            val_accuracies.append(val_correct / len(val_data))

            print("Epoch: {}/{}.. ".format(e + 1, hype["epochs"]))
            print("Training Accuracy: {:.3f}.. ".format(train_accuracies[-1]))
            print("Validation Accuracy: {:.3f}.. ".format(val_accuracies[-1]))
            trial.report(val_accuracies[-1], e)

            if trial.should_prune():
                raise optuna.TrialPruned()

    return val_accuracies[-1]


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3, n_warmup_steps=3, interval_steps=2
        ),
    )
    study.optimize(objective, n_trials=20)
    print("The best parameters are thus: ", study.best_params)
