# -*- coding: utf-8 -*-
import logging

import click

from src.visualization.visualize import (drift_detection,
                                         plot_class_distributions,
                                         plot_tsne_test_set)


@click.command()
@click.argument(
    "trained_model_filepath", type=click.Path(), default="models/trained_model.pth"
)
@click.argument(
    "test_data_filepath", type=click.Path(), default="/data/processed/test.pt"
)
@click.argument(
    "training_data_filepath", type=click.Path(), default="/data/processed/training.pt"
)
@click.argument("figures_folderpath", type=click.Path(), default="reports/figures/")
def main(
    trained_model_filepath,
    training_data_filepath,
    test_data_filepath,
    figures_folderpath,
):
    plot_tsne_test_set(trained_model_filepath, test_data_filepath, figures_folderpath)
    plot_class_distributions(
        training_data_filepath, test_data_filepath, figures_folderpath
    )
    drift_detection(
        training_data_filepath,
        test_data_filepath,
        trained_model_filepath,
        figures_folderpath,
        plot_examples=False,
        detector_num_batches=50,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
