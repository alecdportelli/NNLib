"""
SGD.py

Description: SGD Optimizer

Author: Alec Portelli
Date: 2024-06-04
License: MIT
"""

import numpy as np

from .OptimizerBase import OptimizerBase


class SGD(OptimizerBase):
    def __init__(self, learning_rate:float=1., decay=0., momentum=0.):
        super().__init__(learning_rate, decay)
        self.current_learning_rate = learning_rate
        self.iterations = 0
        self.momentum = momentum

    def before_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1/(1. + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.derivative_weights
            layer.weight_momentums = weight_updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.derivative_biases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.derivative_weights
            bias_updates = -self.current_learning_rate * layer.derivative_biases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def after_update_params(self):
        self.iterations += 1