"""
nn_playground.py

Description: Test file for trying out nn features

Author: Alec Portelli
Date: 2024-06-04
License: MIT
"""

import numpy as np

from src.phugoid_nn.Layers.DenseLayer import Dense
from src.phugoid_nn.Optimizers.SGD import SGD


# Create batch size - typically 32 or 64
BATCH_SIZE = 32

input_layer_data = np.random.randn(BATCH_SIZE, 4)
layer1 = Dense(num_inputs=input_layer_data.shape[1], num_neurons=3, activation="relu")

