"""
Adagrad.py

Description: Adagrad Optimizer

Author: Alec Portelli
Date: 2024-06-04
License: MIT
"""

import numpy as np

from .SGD import SGD


class Adagrad(SGD):
    def __init__(self, learning_rate:float=1., decay=0., epsilon=1e-7):
        super().__init__(learning_rate, decay) # Momentum has default value - not needed
        self.epsilon = epsilon

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.derivative_weights**2
        layer.bias_cache += layer.derivative_biases**2

        layer.weights += -self.current_learning_rate * \
                         layer.derivative_weights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        
        layer.biases += -self.current_learning_rate * \
                        layer.derivative_biases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)