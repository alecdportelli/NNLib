"""
nn_playground.py

Description: Test file for trying out nn features

Author: Alec Portelli
Date: 2024-06-04
License: MIT
"""

import numpy as np

from src.phugoid_nn.Layers.DenseLayer import Dense


BATCH_SIZE = 10

input_layer_data = np.random.randn(BATCH_SIZE, 4)
layer1 = Dense(num_inputs=input_layer_data.shape[1], num_neurons=3, activation="relu")
# NOTE: the column (second value in the input layer MUST match the num inputs for the layer)

output1 = layer1.forward(input_layer_data)
# NOTE: The output shape is the batch size by the amount of neurons in that layer

layer2 = Dense(num_inputs=output1.shape[1], num_neurons=5, activation="relu")
# NOTE: the num inputs must match the num columns in previous layer output
layer2.forward(output1)

print(layer2.output.shape)

# TODO: Create YAML for all of this 