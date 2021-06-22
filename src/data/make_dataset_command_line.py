# -*- coding: utf-8 -*-
import logging

import click

from src.data.MakeDataset import MakeDataset


@click.command()
@click.option(
    "-fu/-np-fu",
    "--force_unzip/--no_force_unzip",
    type=bool,
    default=False,
    help="Force unzip of the data",
)
@click.option(
    "-fd/-no-fd",
    "--force_download/--no_force_download",
    type=bool,
    default=False,
    help="Force download of the data",
)
@click.option(
    "-fp/-no-fp",
    "--force_process/--no_force_process",
    type=bool,
    default=False,
    help="Force process of the data",
)
@click.option(
    "-csv/-no-csv",
    "--add_csv_file/--no_add_csv_file",
    type=bool,
    default=False,
    help="Converts the data to csv",
)
@click.option(
    "-gpi",
    "--generations_per_image",
    type=int,
    default=1,
    help="specifiy the number of images generated per images",
)
def main(
    force_unzip, force_download, force_process, add_csv_file, generations_per_image
):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    make_data = MakeDataset(
        force_process=force_process,
        force_download=force_download,
        force_unzip=force_unzip,
        image_size=128,
        generated_images_per_image=generations_per_image,
        add_csv_file=add_csv_file,
    )
    make_data.make_dataset()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
