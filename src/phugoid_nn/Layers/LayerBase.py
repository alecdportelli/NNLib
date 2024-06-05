"""
example.py

Description: Base class for layers in neural network 

Author: Alec Portelli
Date: 2024-06-05
License: MIT
"""

import numpy as np


class Layer:
    """
    Base class for neural network layers.

    Attributes:
    - num_neurons (int): The number of neurons in the layer.
    - biases (numpy.ndarray): The biases associated with the neurons in the layer.
    """

    def __init__(self, num_neurons: int):
        """
        Initializes the LayerBase object with the specified number of neurons.

        Args:
        - num_neurons (int): The number of neurons in the layer.
        """
        self.num_neurons = num_neurons
        self.biases = np.zeros((1, self.num_neurons))
