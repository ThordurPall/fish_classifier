# -*- coding: utf-8 -*-
import logging

import click

from src.models.train_model import train_model


@click.command()
@click.argument(
    "trained_model_filepath", type=click.Path(), default="models/trained_model.pth"
)
@click.argument(
    "training_statistics_filepath", type=click.Path(), default="data/processed/"
)
@click.argument(
    "training_figures_filepath", type=click.Path(), default="reports/figures/"
)
@click.option(
    "-ua",
    "--use_azure",
    type=bool,
    default=False,
    help="Set True to run on Azure (default=False)",
)
@click.option(
    "-e",
    "--epochs",
    type=int,
    default=30,
    help="Number of training epochs (default=10)",
)
@click.option(
    "-lr",
    "--learning_rate",
    type=float,
    default=0.001,
    help="Learning rate for the PyTorch optimizer (default=0.001)",
)
@click.option(
    "-d",
    "--dropout_p",
    type=float,
    default=0.25,
    help="Dropout rate (default=0.0)",
)
@click.option(
    "-a",
    "--activation",
    type=str,
    default="leaky_relu",
    help="Activation function (default: relu)",
)
def train_model_command_line(
    trained_model_filepath,
    training_statistics_filepath,
    training_figures_filepath,
    use_azure,
    epochs,
    learning_rate,
    dropout_p,
    activation,
):
    """Trains the neural network using MNIST training data"""
    _ = train_model(
        trained_model_filepath,
        training_statistics_filepath,
        training_figures_filepath,
        use_azure,
        epochs,
        learning_rate,
        dropout_p,
        activation,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    train_model_command_line()
