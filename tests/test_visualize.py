# -*- coding: utf-8 -*-
import os.path

from src.data.MakeDataset import MakeDataset
from src.models.train_model import train_model
from src.visualization.visualize import (drift_detection,
                                         plot_class_distributions,
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
        training_statistics_filepath = "data/processed/"
        test_data_filepath = "/data/processed/test.pt"
        train_model(
            trained_model_filepath,
            training_statistics_filepath,
            figures_folderpath,
            epochs=3,
        )

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
        trained_model_filepath = "models/trained_model.pth"
        figures_folderpath = "reports/figures/"
        training_statistics_filepath = "data/processed/"
        training_data_filepath = "/data/processed/training.pt"
        test_data_filepath = "/data/processed/test.pt"
        train_model(
            trained_model_filepath,
            training_statistics_filepath,
            figures_folderpath,
            epochs=3,
        )

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

    def test_drift_detection(self):
        """
        Test that doing drift detection works
        """
        # Make sure that the data and model exist
        make_data = MakeDataset(generated_images_per_image=1)
        make_data.make_dataset()
        trained_model_filepath = "models/trained_model.pth"
        figures_folderpath = "reports/figures/"
        training_statistics_filepath = "data/processed/"
        training_data_filepath = "/data/processed/training.pt"
        test_data_filepath = "/data/processed/test.pt"
        train_model(
            trained_model_filepath,
            training_statistics_filepath,
            figures_folderpath,
            epochs=3,
        )

        # Do the drit detection
        dict = drift_detection(
            training_data_filepath,
            test_data_filepath,
            trained_model_filepath,
            figures_folderpath,
        )

        # Test that the drift detection p-values are returned
        assert 0.0 <= dict["PValuesValidationReal"][0] <= 1.0
        assert 0.0 <= dict["PValuesValidationCorrupt"][0] <= 1.0
        assert 0.0 <= dict["PValuesTestReal"][0] <= 1.0
