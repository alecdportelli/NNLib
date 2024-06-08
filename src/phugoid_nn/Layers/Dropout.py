"""
Dropout.py

Description: Dropout layer class

Author: Alec Portelli
Date: 2024-06-05
License: MIT
"""

import numpy as np

from .LayerBase import Layer

from ..Activations.ReLU import ReLU
from ..Activations.Tanh import Tanh
from ..Activations.Sigmoid import Sigmoid


class Dropout(Layer):
    def __init__(self, rate=0):
        super().__init__()
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate,
                           size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask