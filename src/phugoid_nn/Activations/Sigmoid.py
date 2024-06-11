"""
example.py

Description: Dense layer class

Author: Alec Portelli
Date: 2024-06-05
License: MIT
"""

import numpy as np

from .ActivationBase import Activation


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output
    
    def backward(self, derivatives):
        self.derivative_inputs = derivatives * \
            (1 - self.output) * self.output
    