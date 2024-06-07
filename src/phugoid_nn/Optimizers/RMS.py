"""
Adagrad.py

Description: Root Mean Squared Optimizer

Author: Alec Portelli
Date: 2024-06-04
License: MIT
"""

import numpy as np

from .SGD import SGD


class RMS(SGD):
    def __init__(self, learning_rate:float=1., decay=0., epsilon=1e-7, rho=0.9):
        super().__init__(learning_rate, decay) # Momentum has default value - not needed
        self.epsilon = epsilon
        self.rho = rho

    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.derivative_weights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.derivative_biases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         layer.derivative_weights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.derivative_biases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)