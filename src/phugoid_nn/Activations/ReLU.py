"""
ReLU.py

Description: Dense layer class

Author: Alec Portelli
Date: 2024-06-05
License: MIT
"""

import numpy as np

from .ActivationBase import Activation


class ReLU(Activation):
    """
    Class representing the Rectified Linear Unit (ReLU) activation function.

    Inherits from:
    - Activation

    Methods:
    - forward: Computes the forward pass of the ReLU activation function.
    - backward: Computes the backward pass of the ReLU activation function.
    """
    def __init__(self):
        """
        Initializes the ReLU object.
        """
        super().__init__()

    def forward(self, inputs):
        """
        Computes the forward pass of the ReLU activation function.

        Args:
        - inputs (numpy.ndarray): Input data.

        Returns:
        - numpy.ndarray: Output data after applying ReLU activation.
        """
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, derivatives):
        """
        Computes the backward pass of the ReLU activation function.

        Args:
        - derivatives (numpy.ndarray): Gradients of the loss function with respect to the output of the ReLU activation.

        Returns:
        - numpy.ndarray: Gradients of the loss function with respect to the input of the ReLU activation.
        """
        self.derivative_inputs = derivatives.copy()
        self.derivative_inputs[self.inputs <= 0] = 0

