# -*- coding: utf-8 -*-
import os.path

import pytest
import torch

from src.data.make_dataset import make_dataset
from src.models.train_model import train_model


class TestTraining:

    @pytest.mark.parametrize("epochs,lr", [(1, 0.01), (1, 0.1), (2, 0.001), (3, 0.001)])
    def test_training(self, epochs, lr):
        """
        Test that the training and test MNIST data
        has the correct dimensions
        """

        data_filepath = 'tests/tests_temp'
        trained_model_filepath = data_filepath + '/trained_test_model.pth'
        training_statistics_filepath = data_filepath
        training_figures_filepath = data_filepath
        
        # Make sure the data exists
        train, test = make_dataset(data_filepath)
        
        # Run the entire training procedure
        dict = train_model(data_filepath, trained_model_filepath,
                           training_statistics_filepath,
                           training_figures_filepath,
                           epochs, lr)
        
        # Check that the losses and accuracies have the right length
        assert len(dict['train_losses']) == epochs
        assert len(dict['train_accuracies']) == epochs
        assert len(dict['val_losses']) == epochs
        assert len(dict['val_accuracies']) == epochs
        
        # Test that a model has been saved and can be loaded
        assert os.path.isfile(trained_model_filepath)
        state_dict = torch.load(trained_model_filepath)
        
        # Add a test that this workeds
     
        # Test that the learning curves have been saved
        assert os.path.isfile(os.path.join(training_figures_filepath,
                              'Training_Loss.pdf'))
        assert os.path.isfile(os.path.join(training_figures_filepath,
                              'Training_Accuracy.pdf'))
        

        # Test that losses and accuracies have been saved and can be loaded
        assert os.path.isfile(os.path.join(training_statistics_filepath,
                              'train_val_dict.pickle'))

        # Add a test that this can be loaded

