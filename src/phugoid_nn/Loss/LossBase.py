"""
LossBase.py

Description: Test file for trying out nn features

Author: Alec Portelli
Date: 2024-06-04
License: MIT
"""

import numpy as np


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
    # Regularization loss calculation
    def regularization_loss(self, layer):
        regularization_loss = 0

        if layer.weight_regularizer_L1 > 0:
            regularization_loss += layer.weight_regularizer_L1 * \
                                   np.sum(np.abs(layer.weights))

        if layer.weight_regularizer_L2 > 0:
            regularization_loss += layer.weight_regularizer_L2 * \
                                   np.sum(layer.weights *
                                          layer.weights)

        if layer.bias_regularizer_L1 > 0:
            regularization_loss += layer.bias_regularizer_L1 * \
                                   np.sum(np.abs(layer.biases))

        if layer.bias_regularizer_L2 > 0:
            regularization_loss += layer.bias_regularizer_L2 * \
                                   np.sum(layer.biases *
                                          layer.biases)

        return regularization_loss