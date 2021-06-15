# -*- coding: utf-8 -*-
import json

import pytest
import torch

from src.models.Classifier import Classifier

# Opening the hyperparameters JSON file
f = open('./src/hyperparameters.json',)

# Return JSON object as a dictionary
hype = json.load(f)

class TestModel:
    def test_classifier(self):
        """
        Test that given input with shape [X, 1, 28, 28] that the output of
        the model has shape [X, 9] and the the sum of the exponentials
        is one for each input in the batch
        """
        X = 64

        model = Classifier(hype['num_classes'],
                           hype['filter1_in'],
                           hype['filter1_out'],
                           hype['filter2_out'],
                           hype['filter3_out'],
                           hype['image_height'], hype['image_width'],
                           hype['pad'], hype['stride'],
                           hype['kernel'], hype['pool'],
                           hype['fc_1'], hype['fc_2'])
        x = model.forward(torch.rand(X, hype['rgb'], hype['image_height'],
                                     hype['image_width']))

        # Check that the output from the forward has the correct shape
        assert x.shape == torch.Size([X, hype['num_classes']])

        # Test that the sum of the exponentials of the logits is one
        assert X == int(round(torch.exp(x).sum().item()))
    
    @pytest.mark.parametrize("test_input",
                             [torch.rand(1, 2, 3, hype['image_height'],
                                         hype['image_width']),
                              torch.rand(1, hype['image_height'],
                                         hype['image_width']),
                              torch.rand(1, hype['image_height']),
                              torch.rand(1)])  
    def test_classifier_exception_4D(self, test_input):
        model = Classifier(hype['num_classes'],
                           hype['filter1_in'],
                           hype['filter1_out'],
                           hype['filter2_out'],
                           hype['filter3_out'],
                           hype['image_height'], hype['image_width'],
                           hype['pad'], hype['stride'],
                           hype['kernel'], hype['pool'],
                           hype['fc_1'], hype['fc_2'])

        # Chack that there are batch, channel, width and height dimensions
        with pytest.raises(ValueError, match='Expected input to be a 4D tensor'):
            x = model.forward(test_input)

