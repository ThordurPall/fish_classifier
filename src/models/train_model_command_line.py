# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from src.models.train_model import train_model


@click.command()
@click.argument('trained_model_filepath', type=click.Path(),
                default='models/trained_model.pth')
@click.argument('training_statistics_filepath', type=click.Path(),
                default='data/processed/')
@click.argument('training_figures_filepath', type=click.Path(),
                default='reports/figures/')

def train_model_command_line(trained_model_filepath,
                             training_statistics_filepath,
                             training_figures_filepath,
                             ):
    """ Trains the neural network using MNIST training data """
    _ = train_model(trained_model_filepath,
                    training_statistics_filepath,
                    training_figures_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    train_model_command_line()

