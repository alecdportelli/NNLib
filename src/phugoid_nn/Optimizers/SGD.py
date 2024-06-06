"""
SGD.py

Description: SGD Optimizer

Author: Alec Portelli
Date: 2024-06-04
License: MIT
"""

import numpy as np


class SGD:
    def __init__(self, learning_rate:float=1., decay=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def before_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iterations)

    def update_params(self, layer):
        layer.weights += -self.current_learning_rate * layer.derivative_weights
        layer.biases += -self.current_learning_rate * layer.derivative_biases

    def after_update_params(self):
        self.iterations += 1