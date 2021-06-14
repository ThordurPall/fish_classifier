# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click

from src.data.MakeDataset import MakeDataset


@click.command()
#@click.argument('input_filepath', type=click.Path())
#@click.argument('output_filepath', type=click.Path())
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    make_data = MakeDataset()
    make_data.make_dataset()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

