# -*- coding: utf-8 -*-
from src.data.MakeDataset import MakeDataset
from src.models.evaluate_model import evaluate_model
from src.models.train_model import train_model


class TestEvaluate:
    def test_evaluate_model(self):
        """
        Test that the model evaluation runs and returns the accuracy
        """

        # Make sure that the data and model exist
        make_data = MakeDataset(generated_images_per_image=1)
        make_data.make_dataset()
        trained_model_filepath = "models/trained_model.pth"
        training_statistics_filepath = "data/processed/"
        training_figures_filepath = "reports/figures/"

        train_model(
            trained_model_filepath,
            training_statistics_filepath,
            training_figures_filepath,
            epochs=3,
        )

        # Check that the function returns the test accuracy and
        # that it is between zero and one
        accuracy = evaluate_model()
        assert 0.0 <= accuracy <= 1.0
