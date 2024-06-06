"""
example.py

Description: Dense layer class

Author: Alec Portelli
Date: 2024-06-05
License: MIT
"""

import numpy as np

from .ActivationBase import Activation
from .Softmax import Softmax

from phugoid_nn.Loss.CCE import CCE 


class Softmax_Loss_CCE(Activation):
    def __init__(self):
        super().__init__()
        self.activation = Softmax()
        self.loss = CCE()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, derivatives, y_true):
        samples = len(derivatives)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.derivative_inputs = derivatives.copy()
        self.derivative_inputs[range(samples), y_true] -= 1
        self.derivative_inputs = self.derivative_inputs / samples
