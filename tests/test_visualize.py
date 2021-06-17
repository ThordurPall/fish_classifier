# -*- coding: utf-8 -*-
import os.path

from src.data.MakeDataset import MakeDataset
from src.visualization.visualize import (plot_class_distributions,
                                         plot_tsne_test_set)


class TestVisualize:
    def test_plot_tsne_test_set(self):
        """
        Test that plotting the TSNE works
        """
        # Make sure that the data and model exist
        make_data = MakeDataset(generated_images_per_image=1)
        make_data.make_dataset()
        trained_model_filepath = "models/trained_model.pth"
        figures_folderpath = "reports/figures/"
        test_data_filepath = "/data/processed/test.pt"

        # TSNE embedding of the features for the fish test set
        plot_tsne_test_set(
            trained_model_filepath, test_data_filepath, figures_folderpath
        )

        # Test that the TSNE plot was created
        assert os.path.isfile(os.path.join(figures_folderpath, "TSNE_test_set.pdf"))

    def test_plot_class_distributions(self):
        """
        Test that plotting the training and test classs distributions works
        """
        # Make sure that the data and model exist
        make_data = MakeDataset(generated_images_per_image=1)
        make_data.make_dataset()
        figures_folderpath = "reports/figures/"
        training_data_filepath = "/data/processed/training.pt"
        test_data_filepath = "/data/processed/test.pt"

        # Plot the classs distributions
        plot_class_distributions(
            training_data_filepath, test_data_filepath, figures_folderpath
        )

        # Test that the class distributions got saved
        assert os.path.isfile(
            os.path.join(figures_folderpath, "Training_Class_Distribution.pdf")
        )
        assert os.path.isfile(
            os.path.join(figures_folderpath, "Test_Class_Distribution.pdf")
        )
