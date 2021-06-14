# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click

from src.data.MakeDataset import MakeDataset


@click.command()
@click.option('-fu', '--force_unzip', type=bool, default=False,
              help='Force unzip of the data')
@click.option('-fd', '--force_download', type=bool, default=False,
              help='Force download of the data')
@click.option('-fp', '--force_process', type=bool, default=False,
              help='Force process of the data')
def main(force_unzip, force_download, force_process):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    make_data = MakeDataset(force_process=force_process,
                            force_download=force_download,
                            force_unzip=force_unzip,
                            image_size=128,
                            generated_images_per_image=50)
    make_data.make_dataset()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

