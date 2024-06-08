"""
DenseLayer.py

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
    def __init__(self, num_inputs:int=1, num_neurons:int=1, activation:str = "", 
                 weight_regularizer_L1=0, weight_regularizer_L2=0, 
                 bias_regularizer_L1=0, bias_regularizer_L2=0):
        super().__init__(num_neurons)

        self.num_neurons = num_neurons
        self.num_inputs = num_inputs

        self.weight_regularizer_L1 = weight_regularizer_L1
        self.weight_regularizer_L2 = weight_regularizer_L2

        self.bias_regularizer_L1 = bias_regularizer_L1
        self.bias_regularizer_L2 = bias_regularizer_L2

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

        if self.weight_regularizer_L1 > 0:
            derivative_L1 = np.ones_like(self.weights)
            derivative_L1[self.weights < 0] = -1
            self.derivative_weights += self.weight_regularizer_L1 * \
                derivative_L1
            
        if self.weight_regularizer_L2 > 0:
            self.derivative_weights += 2 * self.weight_regularizer_L2 * \
                self.weights
            
        if self.bias_regularizer_L1 > 0:
            derivative_L1 = np.ones_like(self.biases)
            derivative_L1[self.biases < 0] = -1
            self.derivative_biases += self.bias_regularizer_L1 * \
                self.derivative_biases
            
        if self.bias_regularizer_L2 > 0:
            self.derivative_biases += 2 * self.bias_regularizer_L2 * \
                self.biases

        self.derivative_inputs = np.dot(derivatives, self.weights.T)
