# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from src.data.MakeDataset import MakeDataset
from train_model import train_model


@click.command()
#@click.argument('input_filepath', type=click.Path())
#@click.argument('output_filepath', type=click.Path())
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Training model')
    train_model()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

