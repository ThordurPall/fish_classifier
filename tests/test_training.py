# -*- coding: utf-8 -*-
import os.path
import pickle

import pytest
import torch

from src.data.MakeDataset import MakeDataset
from src.models.train_model import train_model


class TestTraining:
    @pytest.mark.parametrize(
        "epochs,learning_rate", [(1, 0.01), (1, 0.1), (2, 0.001), (3, 0.001)]
    )
    def test_training(self, epochs, learning_rate):
        """
        Test the fish classifier training script
        """
        # Make sure the data exists
        make_data = MakeDataset(generated_images_per_image=1)
        make_data.make_dataset()

        # Run the entire training procedure
        trained_model_filepath = "models/trained_model.pth"
        training_statistics_filepath = "data/processed/"
        training_figures_filepath = "reports/figures/"

        dict = train_model(
            trained_model_filepath,
            training_statistics_filepath,
            training_figures_filepath,
            epochs=epochs,
            learning_rate=learning_rate,
        )

        # Check that the losses and accuracies have the right length
        assert len(dict["train_losses"]) == epochs
        assert len(dict["train_accuracies"]) == epochs
        assert len(dict["val_losses"]) == epochs
        assert len(dict["val_accuracies"]) == epochs

        # Test that a model has been saved and can be loaded
        assert os.path.isfile(trained_model_filepath)
        state_dict = torch.load(trained_model_filepath)
        assert hasattr(state_dict, "values")

        # Test that the learning curves have been saved
        assert os.path.isfile(
            os.path.join(training_figures_filepath, "Training_Loss.pdf")
        )
        assert os.path.isfile(
            os.path.join(training_figures_filepath, "Training_Accuracy.pdf")
        )

        # Test that losses and accuracies have been saved and can be loaded
        dict_path = os.path.join(training_statistics_filepath, "train_val_dict.pickle")
        assert os.path.isfile(dict_path)

        with open(dict_path, "rb") as f:
            dict_load = pickle.load(f)

        assert dict["train_losses"] == dict_load["train_losses"]
        assert dict["train_accuracies"] == dict_load["train_accuracies"]
        assert dict["val_losses"] == dict_load["val_losses"]
        assert dict["val_accuracies"] == dict_load["val_accuracies"]
