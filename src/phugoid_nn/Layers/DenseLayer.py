"""
example.py

Description: Dense layer class

Author: Alec Portelli
Date: 2024-06-05
License: MIT
"""

import numpy as np

from .LayerBase import Layer

from ..Activations.ReLU import ReLU
from ..Activations.Tanh import Tanh
from ..Activations.Sigmoid import Sigmoid


class Dense(Layer):
    def __init__(self, num_inputs:int=1, num_neurons:int=1, activation:str = ""):
        super().__init__(num_neurons)

        # Set the weights
        # The 0.01 is to keep numbers in between -1 and 1
        # assuming a NumPy random Gaussian distrobution
        self.weights = 0.01 * np.random.randn(num_inputs, num_neurons)

        # Turn the string into all lowercase for easier parsing
        # and make it set to an attribute 
        self.activation_type = activation.lower()

        # Set the activation type 
        if self.activation_type == "relu":
            self.activation = ReLU()
        elif self.activation_type == "tanh":
            self.activation = Tanh()
        elif self.activation_type == "sigmoid":
            self.activation = Sigmoid()
        else:
            raise ValueError(f"Unsupported activation type: {self.activation_type}")

    def forward(self, inputs:np.ndarray):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
    
    def backward(self, derivatives:np.ndarray):
        self.derivative_weights = np.dot(self.inputs.T, derivatives)
        self.derivative_biases = np.sum(derivatives, axis=0, keepdims=True)
        self.derivative_inputs = np.dot(derivatives, self.weights.T)
