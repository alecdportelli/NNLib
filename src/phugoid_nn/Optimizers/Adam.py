"""
Adam.py

Description: Adam Optimizer

Author: Alec Portelli
Date: 2024-06-04
License: MIT
"""

import numpy as np

from .SGD import SGD


class Adam(SGD):
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        super().__init__(learning_rate=learning_rate)
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * \
                                layer.weight_momentums + \
                                (1 - self.beta_1) * layer.derivative_weights
        layer.bias_momentums = self.beta_1 * \
                            layer.bias_momentums + \
                            (1 - self.beta_1) * layer.derivative_biases


        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))


        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.derivative_weights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.derivative_biases**2


        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))


        layer.weights += -self.current_learning_rate * \
                        weight_momentums_corrected / \
                        (np.sqrt(weight_cache_corrected) +
                            self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        bias_momentums_corrected / \
                        (np.sqrt(bias_cache_corrected) +
                            self.epsilon)
