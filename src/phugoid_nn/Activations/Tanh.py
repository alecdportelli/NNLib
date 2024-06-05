"""
example.py

Description: Dense layer class

Author: Alec Portelli
Date: 2024-06-05
License: MIT
"""

import numpy as np

from .ActivationBase import Activation


class Tanh(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return np.tanh(inputs)
    