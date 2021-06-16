# -*- coding: utf-8 -*-
import logging

import click

from src.models.evaluate_model import evaluate_model


@click.command()
@click.argument(
    "trained_model_filepath", type=click.Path(), default="models/trained_model.pth"
)
def evaluate_model_command_line(trained_model_filepath):
    """ Trains the neural network using MNIST training data """
    accuracy = evaluate_model(trained_model_filepath)
    logger = logging.getLogger(__name__)
    logger.info(str("Test Accuracy: {:.3f}".format(accuracy)))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    evaluate_model_command_line()
