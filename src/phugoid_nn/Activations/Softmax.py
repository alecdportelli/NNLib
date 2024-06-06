"""
ReLU.py

Description: Dense layer class

Author: Alec Portelli
Date: 2024-06-05
License: MIT
"""

import numpy as np

from .ActivationBase import Activation


class Softmax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        self.inputs = inputs

        """
        Compute the maximum value along each row of a 2D NumPy array, retaining the original array's dimensions.

        Parameters:
        inputs (ndarray): A 2D NumPy array.

        Returns:
        ndarray: A 2D array where each element is the maximum value of the corresponding row in the input array.

        Example:
        >>> import numpy as np
        >>> inputs = np.array([[1, 3, 2], [4, 6, 5], [7, 9, 8]])
        >>> max_along_rows(inputs)
        array([[3],
               [6],
               [9]])
        """
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probs
        return probs
    
    def backward(self, derivatives):
        self.derivative_inputs = np.empty_like(derivatives)

        # Loop through all the gradients and outputs
        for i, (output, derivative) in enumerate(zip(self.output, derivatives)):
            output = output.reshape(-1, 1)
            jacobian = np.diagflat(output) - np.dot(output, output.T)
            self.derivative_inputs[i] = np.dot(jacobian, derivative)
            